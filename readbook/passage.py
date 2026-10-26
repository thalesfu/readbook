"""按小样里试听认可的配置，做任意一段：取台词 → 剧本化 → 合成 → 混音 → MP3 和对照剧本。

认可的配置（"混合版"）：
- 春希：分情绪版的定妆音（moods/refs.yaml），旁白和心里话用"讲述"；
- 雪菜：低温版的单一锚点（refs.yaml），不分情绪；
- Qwen3-TTS Base，温度 0.6，句内变调上限 1.45，最多 5 次，语速拉回目标的 0.85–1.2 倍。

    uv run python -m readbook.passage 2002 "01 第二部-共通" 2010-11-25 2002
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from . import qc, wa2
from .bakeoff import MOOD_DIR, QWEN_BASE, WORK, finish, mood_ref, ok, to48k
from .scriptize import load_cast, scriptize
from .synth import load_yaml, tts_text
from .trial import BOOK, OUT, read_jsonl, write_jsonl
from .voices import SR

# 说话人 → 参考来源
VOICE_SOURCE = {"春希": "moods", "雪菜": "single", "依绪": "single", "武也": "single"}
TYPE = {"旁白": "旁白", "台词": "台词", "心理活动": "心里话"}


def pick_ref(row, cast, single, moods_refs):
    if VOICE_SOURCE.get(row["speaker"]) == "moods":
        return mood_ref(row, moods_refs, cast)
    return single[row["speaker"]]


def main(name: str, route: str, day: str, from_sid: str):
    date = tuple(int(x) for x in day.split("-"))
    cast = load_cast(BOOK / "cast.yaml")
    script = BOOK / "script" / f"{name}.jsonl"
    if not script.exists():
        lines = [l for l in wa2.load_day(route, date) if l.sid >= from_sid]
        raw = []
        for i, l in enumerate(lines):
            d = {**l.__dict__, "idx": i, "voice_code": l.voice_code}
            raw.append(d)
        write_jsonl(OUT / "raw" / f"{name}.jsonl", raw)
        write_jsonl(script, scriptize(raw, cast))
        print("剧本化完成", len(raw), "句", flush=True)
    rows = read_jsonl(script)

    single = yaml.safe_load((WORK / "refs.yaml").read_text(encoding="utf-8"))
    moods_refs = yaml.safe_load((MOOD_DIR / "refs.yaml").read_text(encoding="utf-8"))
    lex = load_yaml(BOOK / "lexicon.yaml")
    out = OUT / "passage" / name
    wav = out / "wav"
    wav.mkdir(parents=True, exist_ok=True)

    from mlx_audio.tts.utils import load_model

    model = None
    log = json.loads((out / "qc.json").read_text()) if (out / "qc.json").exists() else {}
    for r in rows:
        f = wav / f"{r['idx']:04d}.wav"
        text = tts_text(r, lex)
        if f.exists():
            continue
        if not qc.SPOKEN.search(text):
            sf.write(f, np.zeros(int(0.8 * SR), dtype=np.float32), SR)
            continue
        model = model or load_model(QWEN_BASE)
        ref = pick_ref(r, cast, single, moods_refs)
        cands = []
        for _ in range(5):
            res = list(model.generate(text=text, ref_audio=ref["zh_audio"], ref_text=ref["zh_text"],
                                      lang_code="chinese", temperature=0.6))
            a = to48k(np.concatenate([np.array(x.audio, dtype=np.float32) for x in res]), res[0].sample_rate)
            cands.append(a)
            if ok(a, text, ref, 1.45):
                break
        a, m = finish(cands, text, ref)
        sf.write(f, a, SR)
        log[str(r["idx"])] = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}
        print(r["idx"], log[str(r["idx"])], flush=True)
        (out / "qc.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")

    mix(name, rows, cast, wav, out)


def mix(name, rows, cast, wav, out):
    from .mix import assemble, to_mp3

    has_phone = any(r["effect"] == "电话" for r in rows)
    for label, phone in (("", True), ("（无电话效果）", False))[: 2 if has_phone else 1]:
        rs = rows if phone else [{**r, "effect": "无" if r["effect"] == "电话" else r["effect"]} for r in rows]
        marks = []
        audio = assemble(rs, wav, cast, inner=True, marks=marks)
        mp3 = out / f"{name}{label}.mp3"
        to_mp3(audio, mp3, {"album": "白色相簿2 剧本全集（试做）", "title": f"{name}{label}",
                            "artist": "AI 有声书试做"})
        print(mp3, f"{len(audio) / SR / 60:.1f} 分钟", flush=True)
        if phone:
            write_table(name, rows, marks, out)


def write_table(name, rows, marks, out):
    head = [f"# {name} 对照剧本", "",
            "时间戳按带电话效果的那版算，无电话效果版的时间轴相同。", "",
            "| 时间 | # | 谁 | 类别 | 情绪 | 效果 | 台词 |", "|---|---|---|---|---|---|---|"]
    body = []
    for r, t in zip(rows, marks):
        text = r["text"].replace("\n", " ").replace("|", "｜")
        if not qc.SPOKEN.search(text):
            text += "（静音）"
        body.append(f"| {int(t // 60)}:{t % 60:04.1f} | {r['idx']} | {r['speaker']} | {TYPE[r['type']]} "
                    f"| {r.get('mood', '')} | {r['effect']} | {text} |")
    (out / f"{name} 对照剧本.md").write_text("\n".join(head + body) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:5])
