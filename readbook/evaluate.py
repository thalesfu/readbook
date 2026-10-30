"""评测：对任意一个版本的小样或〈2002〉出完整指标表（Markdown）和逐句明细（jsonl）。

    uv run python -m readbook.evaluate 小样 out/bakeoff/qwen-anchor
    uv run python -m readbook.evaluate 2002 "books/wa2/baseline/2010-11-25 〈2002〉-混合版.mp3"
    uv run python -m readbook.evaluate 小样 <MP3> --锚点 春希=books/wa2/trials/rawavg-anchor/锚点/春希.wav

输入两种：
- 逐句 wav 的目录（<idx 四位>.wav，48k 干声，bakeoff / passage 的输出），结果写在这个目录旁边；
- 混好的 MP3，结果写在 MP3 旁边。mix.assemble 在句间插的是纯零，长度已知（0.25/0.4/0.5 秒，只有标点的
  句子另插 0.8 秒），所以能按纯零静音切回逐句。但长句里的停顿有时也是纯零，所以再拿对照剧本的时间戳
  估每句多长，用动态规划挑出最合适的一组句界；对照剧本默认找同目录下名字最像的那份。切出来的句子
  带着混音效果（房间感、回忆、内心处理），所以 MP3 的数字只和 MP3 比。

指标（技术报告第七节，05 号票补强）：
- 像原声：和本人日语原声嵌入中心的 CAM++ 相似度。原声从语音包里按固定种子随机取 300 句（1.5–15 秒），
  向量缓存在 out/eval/real.npz；
- 像锚点：和音色锚点（books/wa2/voices/<名字>/平常.wav，--锚点 <名字>=<wav> 可以换掉某个人的，能写多个）的相似度；
- 音色门槛：本人原声和原声中心相似度的 P5（和逐句分数同一尺度），见 voice_gate；
- 认人准确率：离四个人的锚点哪个最近；另附按原声中心算的；
- 不到 1.5 秒的句子单独统计：声纹模型在短句上测不准；
- 情感 × 音色：emotion2vec+ large 给每句识别情绪，看每种情绪下的相似度；另附按剧本标注情绪分的；
- 识别字错率（Whisper large-v3-turbo）、音高和语速范围、句内变调，沿用以前的做法。
春希的旁白和心里话（讲述模式）单列一组。
"""

import json
import random
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from . import qc
from .asrcheck import PUNCT, TorchWhisper, cer
from .bakeoff import sample_rows
from .config import ROOT, hf_model
from .mix import IR, gap
from .scriptize import load_cast
from .speaker import SpeakerSim
from .synth import SILENCE, load_yaml, tts_text
from .trial import BOOK, OUT, read_jsonl, write_jsonl
from .voices import SR, VoicePak

PASSAGES = {
    "小样": ("小样（2011-01-01，剧本 2024 第 0–13、54–84 句）", sample_rows),
    "2002": ("〈2002〉（2010-11-25，剧本 2002 起 65 句）",
             lambda: read_jsonl(BOOK / "script" / "2010-11-25 〈2002〉.jsonl")),
}
VOICES = BOOK / "voices"
SHORT = 1.5                  # 短于这个秒数（去掉首尾静音）的句子单独统计
REAL_N, REAL_SEED = 300, 0   # 每个角色取多少句原声、随机种子
FLOOR = -85                  # dBFS；句间的纯零在 MP3 解码后低于 -90
EMOTION = "emotion2vec/emotion2vec_plus_large"


# ---------- 取逐句音频 ----------

def spoken(row: dict, lex: dict) -> bool:
    return bool(qc.SPOKEN.search(tts_text(row, lex)))


def read_wav(path: Path) -> np.ndarray:
    a, sr = sf.read(path, dtype="float32")
    assert sr == SR
    return a.mean(axis=1) if a.ndim > 1 else a


def decode(path: Path) -> np.ndarray:
    """MP3 → 48k 单声道。"""
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "f32le", "-ac", "1",
                          "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def find_script(mp3: Path) -> Path | None:
    """同目录下和 MP3 名字共同前缀最长的对照剧本。"""
    def common(p):
        return next((i for i, (x, y) in enumerate(zip(p.stem, mp3.stem)) if x != y),
                    min(len(p.stem), len(mp3.stem)))
    cands = sorted(mp3.parent.glob("*对照剧本.md"), key=common, reverse=True)
    return cands[0] if cands and common(cands[0]) > 0 else None


def read_marks(md: Path) -> dict[int, float]:
    """对照剧本里每句的起始秒数。"""
    marks = {}
    for line in md.read_text(encoding="utf-8").splitlines():
        if m := re.match(r"\|\s*(\d+):(\d+(?:\.\d+)?)\s*\|\s*(\d+)\s*\|", line):
            marks[int(m[3])] = int(m[1]) * 60 + float(m[2])
    return marks


def tail(row: dict) -> float:
    """mix.process 加混响后句尾多出来的秒数（inner=True）。"""
    if row["type"] == "心理活动":
        kind = "inner"
    elif row["effect"] == "回忆":
        kind = "memory"
    elif row["type"] == "台词" and row["effect"] == "无":
        kind = "room"
    else:
        return 0.0
    return len(IR[kind][0]) // 2 / SR


def zero_runs(a: np.ndarray, min_len=0.15) -> tuple[list[tuple[float, float]], float, float]:
    """纯零静音段 [(起, 止)]，以及第一个、最后一个有声帧的时间。按 10ms 一帧算。"""
    hop = SR // 100
    frames = a[: len(a) // hop * hop].reshape(-1, hop)
    quiet = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-12) < FLOOR
    loud = np.flatnonzero(~quiet)
    onset, offset = loud[0] / 100, (loud[-1] + 1) / 100
    edges = np.flatnonzero(np.diff(np.r_[0, quiet.astype(np.int8), 0]))
    runs = [(s / 100, e / 100) for s, e in zip(edges[::2], edges[1::2])
            if e - s >= min_len * 100 and s / 100 > onset and e / 100 < offset]
    return runs, onset, offset


def split(a: np.ndarray, rows: list[dict], lex: dict, marks: dict[int, float]) -> dict[int, tuple[float, float]]:
    """把混好的整段切回逐句：{idx: (起, 止)} 秒。

    每个句界必须落在一段纯零上，而且这段至少和插进去的静音一样长；在满足这一点的组合里，
    挑每句实际长度和预期长度（对数）偏差平方和最小的。预期长度优先用对照剧本的时间戳，
    没有就按每秒 4.2 字估。
    """
    idx = [i for i, r in enumerate(rows) if spoken(r, lex)]
    need, want = [], []
    for k, i in enumerate(idx):
        if k:  # 上一句到这一句之间应有的纯零长度：间隔，加上中间只有标点的句子
            j0 = idx[k - 1]
            need.append(sum(gap(rows[j - 1], rows[j]) + (SILENCE + tail(rows[j]) if j < i else 0)
                            for j in range(j0 + 1, i + 1)))
        r, nxt = rows[i], rows[i + 1] if i + 1 < len(rows) else None
        if nxt and r["idx"] in marks and nxt["idx"] in marks:
            d = marks[nxt["idx"]] - marks[r["idx"]] - gap(r, nxt)
        else:
            d = len(qc.SPOKEN.findall(tts_text(r, lex))) / 4.2 + 0.3 + tail(r)
        want.append(max(d, 0.3))
    runs, onset, offset = zero_runs(a)
    assert len(runs) >= len(idx) - 1, f"纯零静音只有 {len(runs)} 段，少于 {len(idx) - 1} 个句界"
    start_of = lambda r: onset if r < 0 else runs[r][1]  # noqa: E731  这个句界之后那句的起点
    # best[r] = (累计代价, 上一个句界的 r)；r = -1 表示开头
    best, back = {-1: (0.0, None)}, []
    for k in range(1, len(idx)):
        cur = {}
        for r, (s, e) in enumerate(runs):
            if e - s < need[k - 1] - 0.06:
                continue
            excess = 0.5 * max(0.0, e - s - need[k - 1] - 0.15)  # 句子自带的首尾静音会让纯零变长，轻罚
            for r0, (c0, _) in best.items():
                d = s - start_of(r0)
                if r0 >= r or d < 0.15:
                    continue
                c = c0 + np.log(d / want[k - 1]) ** 2 + excess
                if r not in cur or c < cur[r][0]:
                    cur[r] = (c, r0)
        back.append(cur)
        best = cur
    last = min(best, key=lambda r0: best[r0][0] + np.log(max(offset - start_of(r0), 0.15) / want[-1]) ** 2)
    chosen = [last]
    for cur in reversed(back[1:]):
        chosen.append(cur[chosen[-1]][1])
    chosen = [-1] + chosen[::-1]
    ends = [runs[r][0] for r in chosen[1:]] + [offset]
    return {rows[i]["idx"]: (start_of(r), e) for i, r, e in zip(idx, chosen, ends)}


# ---------- 模型 ----------

def real_embeddings(sp: SpeakerSim, cast: dict, names: list[str]) -> dict[str, np.ndarray]:
    """每个角色 REAL_N 句原声的 CAM++ 向量，缓存在 out/eval/real.npz。"""
    cache = OUT / "eval" / "real.npz"
    have = dict(np.load(cache)) if cache.exists() else {}
    missing = [n for n in names if n not in have]
    if missing:
        vp = VoicePak()
        for n in missing:
            embs = [sp.embed_audio(a, SR) for _, a in real_clips(vp, cast["characters"][n]["code"])]
            have[n] = np.array(embs)
            print(n, "原声", len(embs), "句", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **have)
    return {n: have[n] for n in names}


def real_pitch(cast: dict, names: list[str]) -> dict[str, np.ndarray]:
    """每个角色同一批 REAL_N 句原声的音高（qc.f0_median，Hz；测不出的不计），缓存在 out/eval/real-pitch.json。"""
    cache = OUT / "eval" / "real-pitch.json"
    have = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    missing = [n for n in names if n not in have]
    if missing:
        vp = VoicePak()
        for n in missing:
            have[n] = [f for _, a in real_clips(vp, cast["characters"][n]["code"]) if (f := qc.f0_median(a))]
            print(n, "原声音高", len(have[n]), "句", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(have, ensure_ascii=False), encoding="utf-8")
    return {n: np.array(have[n]) for n in names}


def real_clips(vp: VoicePak, code: str, files: list[str] | None = None):
    """(语音文件, 音频)：按固定种子随机取的 REAL_N 句 1.5–15 秒原声，音色门槛和原声分布都用这一批。
    files：只从这些语音文件里取（一个编号拆成几条声线时按声线取，见 voicelines）；默认取这个编号的全部。"""
    files = sorted(files if files is not None else (f for f in vp.index if f.endswith(f"_{code}.OGG")))
    random.Random(REAL_SEED).shuffle(files)
    k = 0
    for f in files:
        a = vp.audio(f)
        if 1.5 <= len(a) / SR <= 15:
            yield f, a
            k += 1
        if k >= REAL_N:
            break


def voice_gate(E: np.ndarray) -> tuple[np.ndarray, dict]:
    """原声嵌入中心，以及两种分布的 P5 和中位数：
    - 中心尺度：每句原声和"其余原声"中心的相似度（留一）。和"合成句像原声中心"是同一个尺度，
      音色门槛取它的 P5（用户定的）；
    - 两两：原声两两之间的相似度，只作参考。和中心比的分数天然比和单句比高，两两的 P5 不能直接拿来卡逐句分数。
    """
    total = E.sum(axis=0)
    rest = total[None] - E
    loo = np.einsum("ij,ij->i", E, rest) / np.linalg.norm(rest, axis=1)
    pair = (E @ E.T)[np.triu_indices(len(E), 1)]
    stats = {"center_p5": np.percentile(loo, 5), "center_med": np.median(loo),
             "pair_p5": np.percentile(pair, 5), "pair_med": np.median(pair)}
    return total / np.linalg.norm(total), {"n": len(E)} | {k: float(v) for k, v in stats.items()}


class Emotion:
    """emotion2vec+ large，9 类：生气、厌恶、恐惧、开心、中立、其他、难过、吃惊、<unk>。"""

    def __init__(self):
        from funasr import AutoModel

        path = hf_model(EMOTION, ["model.pt", "config.yaml", "configuration.json", "tokens.txt"])
        self.model = AutoModel(model=str(path), disable_update=True, log_level="ERROR")

    def __call__(self, a: np.ndarray, sr: int = SR) -> tuple[str, float]:
        import torch
        import torchaudio

        w = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)), sr, 16000)
        res = self.model.generate(w.numpy(), granularity="utterance", extract_embedding=False,
                                  disable_pbar=True)[0]
        k = int(np.argmax(res["scores"]))
        return res["labels"][k].split("/")[0].replace("<unk>", "未知"), round(float(res["scores"][k]), 3)


# ---------- 评测 ----------

def group(rec: dict) -> str:
    return rec["speaker"] if rec["type"] == "台词" else f"{rec['speaker']}·讲述"


def evaluate(rows: list[dict], audio: dict[int, np.ndarray], starts: dict[int, float],
             lex: dict, voices: dict, gates: dict, anchors: dict, sp: SpeakerSim) -> list[dict]:
    names = list(voices)
    whisper, emotion = TorchWhisper(), Emotion()
    recs = []
    for r in rows:
        if r["idx"] not in audio:
            continue
        a, text = audio[r["idx"]], tts_text(r, lex)
        rec = {"idx": r["idx"], "speaker": r["speaker"], "type": r["type"], "mood": r.get("mood", ""),
               "text": r["text"]}
        if r["idx"] in starts:
            rec["start"] = round(starts[r["idx"]], 2)
        rec["dur"] = round(qc.speech_seconds(a), 2)
        rec["emotion"], rec["emotion_score"] = emotion(a)
        if len(PUNCT.sub("", text)) >= 2:  # "………"、"呃…" 这类不查
            hyp = whisper.transcribe_audio(a, SR)
            rec |= {"hyp": hyp, "cer": round(cer(text, hyp), 3)}
        if r["speaker"] in names:
            e = sp.embed_audio(a, SR)
            f, rt = qc.f0_median(a), qc.rate(a, text)
            rec |= {"sim_real": round(float(e @ gates[r["speaker"]][0]), 3),
                    "sim_anchor": round(float(e @ anchors[r["speaker"]]), 3),
                    "near_anchor": max(names, key=lambda n: float(e @ anchors[n])),
                    "near_real": max(names, key=lambda n: float(e @ gates[n][0])),
                    "f0": round(f, 1) if f else None, "rate": round(rt, 2) if rt else None,
                    "jump": round(qc.max_jump(a, voices[r["speaker"]]["定妆音"]["平常"]["f0"]), 2)}
        recs.append(rec)
        print(rec["idx"], rec["speaker"], rec.get("sim_real"), rec.get("cer"), rec["emotion"], flush=True)
    return recs


# ---------- 报告 ----------

def med_min(xs) -> str:
    return f"{np.median(xs):.2f}（{min(xs):.2f}）" if xs else "—"


def spread(xs, fmt) -> str:
    xs = [x for x in xs if x]
    if not xs:
        return "—"
    return f"{np.median(xs):{fmt}}（{np.percentile(xs, 10):{fmt}}–{np.percentile(xs, 90):{fmt}}）"


def ratio(hit: int, n: int) -> str:
    return f"{hit}/{n}（{hit / n:.0%}）" if n else "—"


def report(title: str, src: Path, how: str, recs: list[dict], voices: dict, gates: dict,
           anchors: dict, swap: dict[str, Path] | None = None) -> str:
    names = list(voices)
    voiced = [r for r in recs if r["speaker"] in names]
    groups = [g for n in names for g in (n, f"{n}·讲述") if any(group(r) == g for r in voiced)]
    long_ = [r for r in voiced if r["dur"] >= SHORT]
    short = [r for r in voiced if r["dur"] < SHORT]
    gate = {n: gates[n][1]["center_p5"] for n in names}
    out = [f"# {src.stem} 指标", "",
           f"- 片段：{title}",
           f"- 版本：`{rel(src)}`",
           f"- 音频：{how}",
           *([f"- 锚点：`{rel(VOICES)}/<名字>/平常.wav`；" + "，".join(f"{n}换成 `{rel(p)}`" for n, p in swap.items())]
             if swap else []),
           f"- 生成：{date.today()}，`uv run python -m readbook.evaluate`",
           "- 相似度都是 CAM++ 余弦相似度。像原声＝和本人日语原声嵌入中心比；像锚点＝和音色锚点（平常）比。",
           f"- 不到 {SHORT} 秒的句子（去掉首尾静音后）声纹测不准，单列，不进主表。", ""]

    out += ["## 音色门槛", "",
            f"每个角色从语音包里随机取 {REAL_N} 句原声（1.5–15 秒，种子 {REAL_SEED}）。"
            "门槛＝每句原声和其余原声中心相似度的 P5，和逐句的像原声同一尺度；"
            "两两＝原声两两之间的相似度，只作参考。", "",
            "| 角色 | 原声句数 | 门槛 | 原声像中心 中位 | 两两 P5 | 两两中位 | 锚点像原声中心 |",
            "|---|---|---|---|---|---|---|"]
    for n in names:
        s = gates[n][1]
        out.append(f"| {n} | {s['n']} | {s['center_p5']:.2f} | {s['center_med']:.2f} | {s['pair_p5']:.2f} "
                   f"| {s['pair_med']:.2f} | {float(anchors[n] @ gates[n][0]):.2f} |")

    def voice_rows(rs, cols):
        lines = []
        for g in groups + ["合计"]:
            sel = rs if g == "合计" else [r for r in rs if group(r) == g]
            if sel:
                lines.append(f"| {g} | {len(sel)} | " + " | ".join(c(sel) for c in cols) + " |")
        return lines

    cols = [lambda s: med_min([r["sim_real"] for r in s]),
            lambda s: ratio(sum(r["sim_real"] >= gate[r["speaker"]] for r in s), len(s)),
            lambda s: med_min([r["sim_anchor"] for r in s]),
            lambda s: ratio(sum(r["near_anchor"] == r["speaker"] for r in s), len(s)),
            lambda s: ratio(sum(r["near_real"] == r["speaker"] for r in s), len(s))]
    out += ["", f"## 音色（≥{SHORT} 秒）", "",
            "| 组 | 句数 | 像原声 中位（最低） | 过门槛 | 像锚点 中位（最低） | 认人·锚点 | 认人·原声中心 |",
            "|---|---|---|---|---|---|---|"]
    out += voice_rows(long_, cols)
    out += ["", f"## 短句（<{SHORT} 秒，仅供参考）", "",
            "| 组 | 句数 | 像原声 中位（最低） | 过门槛 | 像锚点 中位（最低） | 认人·锚点 | 认人·原声中心 |",
            "|---|---|---|---|---|---|---|"]
    out += voice_rows(short, cols) if short else ["| — | 0 | | | | | |"]
    out += ["", f"认人准确率（全部句子，含短句）：锚点 "
            f"{ratio(sum(r['near_anchor'] == r['speaker'] for r in voiced), len(voiced))}，原声中心 "
            f"{ratio(sum(r['near_real'] == r['speaker'] for r in voiced), len(voiced))}。"]

    out += ["", "## 读音和韵律", "",
            "字错率：Whisper large-v3-turbo 识别回文字，对照送进 TTS 的文本（已按读音表替换），不到 2 个字的不算。"
            "句内变调：半秒一段估音高，最高一段超过目标（定妆音平常的音高）1.45 倍。", "",
            "| 组 | 句数 | 字错率 中位 | 字错率 >0.3 | 音高 Hz 中位（P10–P90） | 语速 字/秒 中位（P10–P90） | 句内变调 |",
            "|---|---|---|---|---|---|---|"]
    for g in groups + ["合计"]:
        sel = voiced if g == "合计" else [r for r in voiced if group(r) == g]
        cers = [r["cer"] for r in sel if r.get("cer") is not None]
        f0 = "—" if g == "合计" else spread([r["f0"] for r in sel], ".0f")
        rt = "—" if g == "合计" else spread([r["rate"] for r in sel], ".1f")
        cm = f"{np.median(cers):.3f}" if cers else "—"
        out.append(f"| {g} | {len(sel)} | {cm} | {sum(x > 0.3 for x in cers)} | {f0} | {rt} "
                   f"| {sum(r['jump'] > 1.45 for r in sel)} |")
    others = [r for r in recs if r["speaker"] not in names and r.get("cer") is not None]
    if others:
        out.append(f"\n其他说话人 {len(others)} 句，字错率中位 {np.median([r['cer'] for r in others]):.3f}。")

    for key, label in (("emotion", " emotion2vec+ 识别的情绪"), ("mood", "剧本标注的情绪")):
        cats = sorted({r[key] for r in long_}, key=lambda m: -sum(r[key] == m for r in long_))
        out += ["", f"## 情感 × 音色：按{label}", "",
                f"格子里是像原声 中位（句数），只算 ≥{SHORT} 秒的句子；最低值和门槛的对比见逐句表。", "",
                "| 情绪 | " + " | ".join(groups) + " |", "|---" * (len(groups) + 1) + "|"]
        for m in cats:
            cells = []
            for g in groups:
                xs = [r["sim_real"] for r in long_ if group(r) == g and r[key] == m]
                cells.append(f"{np.median(xs):.2f}（{len(xs)}）" if xs else "")
            out.append(f"| {m} | " + " | ".join(cells) + " |")

    bad = [r for r in recs if (r.get("cer") or 0) > 0.3]
    if bad:
        out += ["", "## 字错率 >0.3 的句子", "", "| # | 谁 | 字错率 | 原文 | 识别 |", "|---|---|---|---|---|"]
        for r in bad:
            out.append(f"| {r['idx']} | {r['speaker']} | {r['cer']:.2f} | {cell(r['text'])} | {cell(r['hyp'])} |")

    timed = any("start" in r for r in recs)
    out += ["", "## 逐句", "",
            "像原声低于门槛的标 ✗；* 是不到 1.5 秒的短句。最近＝离哪个人的锚点最近。", "",
            ("| 时间 " if timed else "") + "| # | 谁 | 类别 | 剧本情绪 | 识别情绪 | 秒 | 像原声 | 像锚点 | 最近 | 字错率 | 台词 |",
            ("|---" if timed else "") + "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        t = f"| {int(r['start'] // 60)}:{r['start'] % 60:04.1f} " if timed else ""
        sim = anc = near = ""
        if "sim_real" in r:
            sim = f"{r['sim_real']:.2f}" + (" ✗" if r["sim_real"] < gate[r["speaker"]] else "")
            anc, near = f"{r['sim_anchor']:.2f}", r["near_anchor"]
        dur = f"{r['dur']:.1f}" + ("*" if r["dur"] < SHORT else "")
        c = f"{r['cer']:.2f}" if r.get("cer") is not None else ""
        out.append(f"{t}| {r['idx']} | {r['speaker']} | {r['type']} | {r['mood']} | {r['emotion']} | {dur} "
                   f"| {sim} | {anc} | {near} | {c} | {cell(r['text'])} |")
    return "\n".join(out) + "\n"


def rel(p: Path) -> str:
    return (p.resolve().relative_to(ROOT) if p.resolve().is_relative_to(ROOT) else p).as_posix()


def cell(text: str) -> str:
    return text.replace("\n", " ").replace("|", "｜")


def main(passage: str, src: str, script: str | None = None, swap: dict[str, Path] | None = None):
    """swap：{名字: wav}，换掉这些人的音色锚点（像锚点、认人·锚点跟着变）。"""
    title, load_rows = PASSAGES[passage]
    src = Path(src)
    rows = load_rows()
    lex = load_yaml(BOOK / "lexicon.yaml")
    cast = load_cast(BOOK / "cast.yaml")
    voices = yaml.safe_load((VOICES / "voices.yaml").read_text(encoding="utf-8"))
    if src.is_dir():
        audio = {r["idx"]: read_wav(f) for r in rows
                 if spoken(r, lex) and (f := src / f"{r['idx']:04d}.wav").exists()}
        starts, how, stem = {}, f"逐句干声，{len(audio)} 句", src.parent / f"{src.name}-指标"
    else:
        script = Path(script) if script else find_script(src)
        a = decode(src)
        spans = split(a, rows, lex, read_marks(script) if script else {})
        audio = {i: a[int(t0 * SR):int(t1 * SR)] for i, (t0, t1) in spans.items()}
        starts = {i: t0 for i, (t0, _) in spans.items()}
        how = (f"从混好的 MP3 按句间纯零切回逐句，{len(audio)} 句（预期长度参考 "
               f"{f'`{script.name}`' if script else '字数'}），带混音效果，只和 MP3 的数字比")
        stem = src.with_name(f"{src.stem}-指标")

    sp = SpeakerSim()
    names = list(voices)
    real = real_embeddings(sp, cast, names)
    gates = {n: voice_gate(E) for n, E in real.items()}
    anchors = {n: sp.embed((swap or {}).get(n, VOICES / n / "平常.wav")) for n in names}
    recs = evaluate(rows, audio, starts, lex, voices, gates, anchors, sp)
    write_jsonl(stem.with_name(stem.name + "-逐句.jsonl"), recs)
    md = stem.with_name(stem.name + ".md")
    md.write_text(report(title, src, how, recs, voices, gates, anchors, swap), encoding="utf-8", newline="\n")
    print(md)


if __name__ == "__main__":
    args, swap = [], {}
    it = iter(sys.argv[1:])
    for a in it:
        if a == "--锚点":
            n, _, p = next(it).partition("=")
            swap[n] = Path(p)
        else:
            args.append(a)
    main(*args[:3], swap=swap)
