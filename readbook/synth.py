"""逐句合成：VoxCPM2（MLX）负责复刻与合成，Qwen3-TTS VoiceDesign 负责给龙套设计音色。

每句产出一个 48kHz 单声道 wav，缓存在 out/wav/<版本>/<片段>/<idx>.wav，已存在就跳过。
版本：
- post：不给指令，心理活动留给混音做后期处理；
- instruct：每句带上 LLM 的表演提示，心理活动额外要求轻声自语，混音不做内心处理；
- orig：在 post 的基础上，有原声的台词把这一句的日语原声当作续写提示，用来带情绪。
"""

import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

SR = 48000
VOXCPM = "mlx-community/VoxCPM2-bf16"
QWEN_DESIGN = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
POOL_SAMPLE = "今天的天气不错，我们一起出去走走吧。这件事情说来话长，你先坐下，听我慢慢讲。"


SPOKEN = re.compile(r"[\w]", re.U)
SILENCE = 0.8


class RunawayError(RuntimeError):
    pass


def max_seconds(text: str) -> float:
    """正常语速约每字 0.2–0.3 秒；超过这个上限就当作生成失控。"""
    return 2.5 + 0.5 * len(SPOKEN.findall(text))


def truncate(audio: np.ndarray, seconds: float, fade: float = 0.3) -> np.ndarray:
    n = min(len(audio), int(seconds * SR))
    out = audio[:n].copy()
    k = min(n, int(fade * SR))
    out[n - k:] *= np.linspace(1, 0, k, dtype=np.float32)
    return out


@dataclass
class Voice:
    ref_audio: str
    ref_text: str | None = None


def tts_text(row: dict, lexicon: dict[str, str]) -> str:
    """台词去掉外层引号；全角字母数字转半角（夹在中间的全角空格一起转，否则 TTS 会把它读成怪音，
    比如「Ｈａｐｐｙ　Ｎｅｗｙｅａｒ」）；套读音表；换行合并。"""
    t = re.sub(r"[Ａ-Ｚａ-ｚ０-９]+(?:　[Ａ-Ｚａ-ｚ０-９]+)*", lambda m: unicodedata.normalize("NFKC", m.group()),
               row["text"].replace("\n", ""))
    if row["type"] == "台词":
        t = re.sub(r"^[“‘](.*)[”’]$", r"\1", t)
    for word, say in lexicon.items():
        t = t.replace(word, say)
    return t


def pool_slot(c: dict) -> str:
    """龙套按性别年龄挑音色池里的一个。"""
    young = c.get("age", 30) < 35
    if c.get("gender") == "女":
        return "女青年A" if young else "女中年A"
    return "男青年A" if young else "男中年A"


class Synth:
    def __init__(self, cast: dict, refs: dict, lexicon: dict, root: Path):
        from mlx_audio.tts.utils import load_model

        self.cast, self.lexicon, self.root = cast, lexicon, root
        self.model = load_model(VOXCPM)
        self.voices: dict[str, Voice] = {}
        for name, c in cast["characters"].items():
            if name in refs:
                r = refs[name]
                self.voices[name] = Voice(str(root / r["ref_audio"]), r["ref_text"])
        self.pool_dir = root / "out" / "refs" / "pool"
        self.stats = {"audio": 0.0, "wall": 0.0}
        self.fallbacks: list[str] = []

    def design_pool(self, names: set[str]):
        """为用到的龙套音色池条目各生成一段中文样音，当作复刻参考。"""
        need = {pool_slot(self.cast["characters"][n]) for n in names if n not in self.voices}
        todo = [s for s in need if not (self.pool_dir / f"{s}.wav").exists()]
        if todo:
            from mlx_audio.tts.utils import load_model

            self.pool_dir.mkdir(parents=True, exist_ok=True)
            designer = load_model(QWEN_DESIGN)
            for slot in todo:
                res = list(designer.generate_voice_design(
                    text=POOL_SAMPLE, language="Chinese", instruct=self.cast["pool"][slot]))
                audio = np.concatenate([np.array(r.audio) for r in res])
                sf.write(self.pool_dir / f"{slot}.wav", audio, res[0].sample_rate)
            del designer
        for n in names:
            if n not in self.voices:
                slot = pool_slot(self.cast["characters"][n])
                self.voices[n] = Voice(str(self.pool_dir / f"{slot}.wav"), POOL_SAMPLE)

    def generate(self, text: str, voice: Voice, instruct: str | None = None,
                 prompt: tuple[str, str] | None = None) -> np.ndarray:
        kw = {"ref_audio": voice.ref_audio}
        if instruct:
            kw["instruct"] = instruct
        if prompt:
            kw["prompt_audio"], kw["prompt_text"] = prompt
        t0 = time.perf_counter()
        parts = [np.array(r.audio, dtype=np.float32) for r in self.model.generate(text=text, **kw)]
        audio = np.concatenate(parts)
        self.stats["wall"] += time.perf_counter() - t0
        self.stats["audio"] += len(audio) / SR
        return audio

    def line(self, row: dict, version: str, orig_dir: Path | None = None) -> np.ndarray:
        voice = self.voices[row["speaker"]]
        text = tts_text(row, self.lexicon)
        if not SPOKEN.search(text):  # "………" 这类只有标点的句子是沉默，不送去合成
            return np.zeros(int(SILENCE * SR), dtype=np.float32)
        instruct = prompt = None
        if version == "instruct":
            instruct = row["instruct"]
            if row["type"] == "心理活动":
                instruct = f"压低声音轻声自语，{instruct}"
        elif version == "orig" and row["type"] == "台词" and row.get("voice") and orig_dir:
            prompt = (str(orig_dir / row["voice"].replace(".OGG", ".wav")), row["jp"].replace("\n", ""))
        # 失控保护：先换种子重试，再去掉指令/提示；都不行由调用方退回 post 版
        tries = []
        for kw in ({"instruct": instruct, "prompt": prompt}, {"instruct": instruct, "prompt": prompt},
                   {}):
            audio = self.generate(text, voice, **kw)
            if len(audio) / SR <= max_seconds(text):
                return audio
            tries.append(audio)
        raise RunawayError(f"#{row['idx']} 生成失控：{text[:20]}", min(tries, key=len))

    def run(self, rows: list[dict], version: str, out: Path, orig_dir: Path | None = None,
            share: Path | None = None):
        """逐句合成。share 指向 post 版目录时，orig 版里没用上原声的句子直接复用。"""
        out.mkdir(parents=True, exist_ok=True)
        for row in rows:
            f = out / f"{row['idx']:04d}.wav"
            if f.exists():
                continue
            uses_orig = version == "orig" and row["type"] == "台词" and row.get("voice")
            if version == "orig" and not uses_orig and share and (share / f.name).exists():
                f.write_bytes((share / f.name).read_bytes())
                continue
            try:
                sf.write(f, self.line(row, version, orig_dir), SR)
            except RunawayError as e:
                self.fallbacks.append(f"{version} #{row['idx']}")
                if share and share != out and (share / f.name).exists():
                    print(f"{version} {e.args[0]}，退回 post 版", flush=True)
                    f.write_bytes((share / f.name).read_bytes())
                else:  # post 版自己没得退：取最短的一次，截到上限的 1.5 倍并淡出
                    print(f"{version} {e.args[0]}，截断", flush=True)
                    sf.write(f, truncate(e.args[1], max_seconds(tts_text(row, self.lexicon)) * 1.5), SR)


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
