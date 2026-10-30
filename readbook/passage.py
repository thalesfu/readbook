"""按小样里试听认可的配置，做任意一段：取台词 → 剧本化 → 合成 → 混音 → MP3 和对照剧本。

认可的配置（"混合版"），定妆音见 books/wa2/voices/voices.yaml：
- 春希：分情绪版的定妆音，旁白和心里话用"讲述"；
- 雪菜：低温版的单一锚点，不分情绪；
- 依绪、武也：锚点版，情绪定妆音从锚点派生；逐句多一道 CAM++ 门槛（和锚点 ≥0.65），最多 6 次；
- Qwen3-TTS Base，温度 0.6，句内变调上限 1.45，最多 5 次，语速拉回目标的 0.85–1.2 倍。
Mac 上用 mlx-audio 版，Windows 上用官方 PyTorch 版（见 bakeoff.TorchQwen）。

07 号票的三个对照版本，参考都换成日语原声（见 rawref），其余照旧；依绪、武也的逐句门槛改成和本人原声中心比、
过音色门槛。产物在 out/passage/<名字>-<版本>/。
- --原声版：带文本复刻，参考是原声加日文逐字稿（有日语口音，不可用）；
- --原声向量版：只取拼好的那段原声的说话人向量，参考码和逐字稿不进上下文；
- --原声平均版：说话人向量取几十句原声的平均。

    uv run python -m readbook.passage 小样 [--原声版|--原声向量版|--原声平均版]
    uv run python -m readbook.passage "2010-11-25 〈2002〉" "01 第二部-共通" 2010-11-25 2002
    uv run python -m readbook.passage remix 小样 <wav 目录> <输出目录> [名字]   # 只重新混音
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from . import qc, wa2
from .bakeoff import anchor_ref, load_qwen, mood_ref, sample_rows, synth_anchor, synth_qc
from .scriptize import load_cast, pauses, scriptize, with_pauses
from .synth import load_yaml, tts_text
from .trial import BOOK, OUT, read_jsonl, write_jsonl
from .voices import SR

VOICES = BOOK / "voices"
# 说话人 → 参考来源
VOICE_SOURCE = {"春希": "moods", "雪菜": "single", "依绪": "anchor", "武也": "anchor"}
TYPE = {"旁白": "旁白", "台词": "台词", "心理活动": "心里话"}
VARIANTS = ("混合版", "原声版", "原声向量版", "原声平均版")


def load_voices() -> dict:
    """voices.yaml → {角色: {情绪: 参考}}，音频换成绝对路径。"""
    data = yaml.safe_load((VOICES / "voices.yaml").read_text(encoding="utf-8"))
    return {n: {m: {"mood": m, "zh_audio": str(VOICES / v["audio"]), "zh_text": v["text"],
                    "f0": v["f0"], "rate": v["rate"]} for m, v in c["定妆音"].items()}
            for n, c in data.items()}


def load_refs(variant: str) -> dict:
    from . import rawref

    return load_voices() if variant == "混合版" else rawref.load(variant)


def variant_arg(args: list[str]) -> tuple[list[str], str]:
    """命令行里的 --<版本>；没有就是混合版。"""
    flags = [a[2:] for a in args if a.startswith("--")]
    assert all(f in VARIANTS for f in flags), f"版本只能是 {VARIANTS}"
    return [a for a in args if not a.startswith("--")], flags[0] if flags else "混合版"


def pick_ref(row, cast, voices):
    n = row["speaker"]
    src = VOICE_SOURCE.get(n)
    if src == "moods":
        return mood_ref(row, {n: {**voices[n], "_have": sorted(voices[n])}}, cast)
    if src == "anchor":
        return anchor_ref(row, voices, cast)
    return voices[n]["平常"]


def load_rows(name: str, route: str | None, day: str | None, from_sid: str | None) -> list[dict]:
    if name == "小样":
        return sample_rows()
    cast = load_cast(BOOK / "cast.yaml")
    script = BOOK / "script" / f"{name}.jsonl"
    if not script.exists():
        date = tuple(int(x) for x in day.split("-"))
        lines = [l for l in wa2.load_day(route, date) if l.sid >= from_sid]
        raw = []
        for i, l in enumerate(lines):
            d = {**l.__dict__, "idx": i, "voice_code": l.voice_code}
            raw.append(d)
        write_jsonl(OUT / "raw" / f"{name}.jsonl", raw)
        rows = scriptize(raw, cast)
        write_jsonl(script, with_pauses(rows, pauses(rows, cast)))
        print("剧本化完成", len(raw), "句", flush=True)
    return read_jsonl(script)


def raw_gates(sp) -> dict:
    """原声版的逐句门槛：{角色: (原声嵌入中心, 音色门槛)}，和 evaluate 同一套。"""
    from .evaluate import real_embeddings, voice_gate

    cast = load_cast(BOOK / "cast.yaml")
    real = real_embeddings(sp, cast, [n for n, s in VOICE_SOURCE.items() if s == "anchor"])
    return {n: (c, s["center_p5"]) for n, (c, s) in ((n, voice_gate(E)) for n, E in real.items())}


def main(name: str, route: str | None = None, day: str | None = None, from_sid: str | None = None,
         variant: str = "混合版"):
    rows = load_rows(name, route, day, from_sid)
    cast = load_cast(BOOK / "cast.yaml")
    voices = load_refs(variant)
    raw = variant != "混合版"
    lex = load_yaml(BOOK / "lexicon.yaml")
    label = f"{name}-{variant}" if raw else name
    out = OUT / "passage" / label
    wav = out / "wav"
    wav.mkdir(parents=True, exist_ok=True)

    model = sp = gates = None
    log = json.loads((out / "qc.json").read_text(encoding="utf-8")) if (out / "qc.json").exists() else {}
    for r in rows:
        f = wav / f"{r['idx']:04d}.wav"
        text = tts_text(r, lex)
        if f.exists():
            continue
        if not qc.SPOKEN.search(text):
            sf.write(f, np.zeros(int(0.8 * SR), dtype=np.float32), SR)
            continue
        model = model or load_qwen()
        ref = pick_ref(r, cast, voices)
        t = time.perf_counter()
        if VOICE_SOURCE.get(r["speaker"]) == "anchor":
            if sp is None:
                from .speaker import SpeakerSim

                sp = SpeakerSim()
            if raw:
                gates = gates or raw_gates(sp)
                a, m = synth_anchor(model, sp, text, ref, target=gates[r["speaker"]][0],
                                    line=gates[r["speaker"]][1])
            else:
                a, m = synth_anchor(model, sp, text, ref)
        else:
            a, m = synth_qc(model, text, ref)
        m.update(ref=ref["mood"], secs=round(time.perf_counter() - t, 2), dur=round(len(a) / SR, 2))
        sf.write(f, a, SR)
        log[str(r["idx"])] = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}
        print(r["idx"], log[str(r["idx"])], flush=True)
        (out / "qc.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")

    mix(label, rows, cast, wav, out)


def mix(name, rows, cast, wav, out, highs: str | None = None) -> list[float]:
    """返回每句的起始秒数（两版时间轴相同）。highs：mix.HIGHS 里的一档，None 用默认。"""
    from .mix import assemble, to_mp3

    has_phone = any(r["effect"] == "电话" for r in rows)
    for label, phone in (("", True), ("（无电话效果）", False))[: 2 if has_phone else 1]:
        rs = rows if phone else [{**r, "effect": "无" if r["effect"] == "电话" else r["effect"]} for r in rows]
        marks = []
        audio = assemble(rs, wav, cast, inner=True, marks=marks, highs=highs)
        mp3 = out / f"{name}{label}.mp3"
        to_mp3(audio, mp3, {"album": "白色相簿2 剧本全集（试做）", "title": f"{name}{label}",
                            "artist": "AI 有声书试做"})
        print(mp3, f"{len(audio) / SR / 60:.1f} 分钟", flush=True)
        if phone:
            write_table(name, rows, marks, out)
            starts = marks
    return starts


def remix(name: str, wav: Path, out: Path, label: str, pause: dict[int, float] | None = None,
          highs: str | None = None) -> list[float]:
    """只重新混音：拿现成的逐句 wav 照剧本再混一遍，不合成。pause 给了就盖掉剧本里的 pause。"""
    rows = load_rows(name, None, None, None)
    if pause is not None:
        rows = [{**r, "pause": pause.get(r["idx"], 0)} for r in rows]
    out.mkdir(parents=True, exist_ok=True)
    return mix(label, rows, load_cast(BOOK / "cast.yaml"), wav, out, highs)


def stamp(t: float) -> str:
    """秒数 → m:ss.s。先取整到 0.1 秒再拆分钟，免得 299.97 写成 4:60.0。"""
    m, d = divmod(round(t * 10), 600)
    return f"{m}:{d // 10:02d}.{d % 10}"


def write_table(name, rows, marks, out):
    head = [f"# {name} 对照剧本", "",
            "时间戳按带电话效果的那版算，无电话效果版的时间轴相同。", ""]
    if any(r.get("pause") for r in rows):
        head += ["【前停 N 秒】：这句前面隐去了别人的话，混音时在这句前面留出 N 秒空白（剧本的 pause）。", ""]
    head += ["| 时间 | # | 谁 | 类别 | 情绪 | 效果 | 台词 |", "|---|---|---|---|---|---|---|"]
    body = []
    for r, t in zip(rows, marks):
        text = r["text"].replace("\n", " ").replace("|", "｜")
        if not qc.SPOKEN.search(text):
            text += "（静音）"
        if r.get("pause"):
            text = f"【前停 {r['pause']:g} 秒】{text}"
        body.append(f"| {stamp(t)} | {r['idx']} | {r['speaker']} | {TYPE[r['type']]} "
                    f"| {r.get('mood', '')} | {r['effect']} | {text} |")
    (out / f"{name} 对照剧本.md").write_text("\n".join(head + body) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    if sys.argv[1] == "remix":
        a = sys.argv[2:]
        remix(a[0], Path(a[1]), Path(a[2]), a[3] if len(a) > 3 else a[0])
    else:
        args, variant = variant_arg(sys.argv[1:])
        main(*args[:4], variant=variant)
