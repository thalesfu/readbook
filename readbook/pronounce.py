"""读音（生产 03 号票）：读音表的拼音标注，和按带声调的拼音比读得对不对。

读音不进逐句挑选（合成规格第六节），从源头修：读音表 books/wa2/lexicon.yaml（换成同音字，或者写 IndexTTS-2.5
的拼音标注 <字|拼音声调>）加文本规范化（synth.tts_text）。这里给评测（evaluate）和复核清单（index_passage）
兜一道：Whisper 识别回的文字和送进 TTS 的文本都转成带声调的拼音（pypinyin 的 TONE3，轻声记 5），按音节算
编辑距离，除以原文的音节数，叫拼音错率。
- 同音字不算错：「依绪」识别成「依序」都是 yi1 xu4；声调错算错：「雪」xue3 读成「学」xue2；
- 原文里标注过的字按标注注音（<雪|XVE3> 记 xue3），读音表换过字的按换后的字（小木增）；识别结果也套一遍
  读音表，Whisper 写成原作的「小木曾」也按 zēng 比；
- 标点、空白不算；不是汉字的（字母、数字）一个字符算一个，字母不分大小写。

    uv run python -m readbook.pronounce 检查    # 读音表的拼音标注是不是都在 IndexTTS 的拼音表里
    uv run python -m readbook.pronounce 对比 2.5-表演B混音-按轮 2.5-表演B混音-读音 小样 "2010-11-25 〈2002〉" \\
        --旧候选 2.5-表演B混音    # 03 号票的新旧对比
"""

import json
import re
import sys
from pathlib import Path

from pypinyin import Style, lazy_pinyin

from .config import INDEX_DIR

MARK = re.compile(r"<([^<>|]+)\|([^<>]+)>")  # 拼音标注 <字|拼音声调>，写法同 IndexTTS-2.5
MISREAD = 0.3  # 拼音错率超过它就进复核清单


def plain(text: str) -> str:
    """去掉拼音标注，只留字。"""
    return MARK.sub(r"\1", text)


def syllables(text: str) -> list[str]:
    """带声调的拼音，一个字一个，标点、空白不算。标注过的字用标注的读音（IndexTTS 把 ü 写成 V，jqx 后面
    换回 pypinyin 的 u：XVE3 → xue3；lü 两边都是 lv）。"""
    parts = MARK.split(text)  # [字, 标注的字, 拼音, 字, …]
    flat, say = "", {}
    for k in range(0, len(parts), 3):
        flat += parts[k]
        if k + 1 < len(parts):
            say |= {len(flat) + i: py for i, py in enumerate(parts[k + 2].split())}
            flat += parts[k + 1]
    out = lazy_pinyin(flat, style=Style.TONE3, neutral_tone_with_five=True, errors=list)
    assert len(out) == len(flat), text  # errors=list：不是汉字的也一个字符一个，位置才对得上标注
    out = [re.sub(r"^([jqx])v", r"\1u", say[i].lower()) if i in say else s.lower() for i, s in enumerate(out)]
    return [s for s in out if re.fullmatch(r"\w+", s)]


def misread(ref: str, hyp: str, lex: dict) -> float:
    """拼音错率：识别结果 hyp 和送进 TTS 的文本 ref（可带标注）的音节编辑距离 ÷ ref 的音节数。
    hyp 也套一遍读音表 lex：Whisper 写出原作的写法（小木曾）时，按读音表的读法（zēng）比，不算错。"""
    from .synth import tts_text

    a, b = syllables(ref), syllables(tts_text({"text": hyp, "type": "旁白"}, lex))
    if not a:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1] / len(a)


def unknown(lex: dict) -> list[str]:
    """读音表里 IndexTTS 不认的标注：拼音不在它的 checkpoints/pinyin.vocab 里（比如把 XVE3 写成 XUE3），
    模型会把它当成几个不相干的字母。"""
    vocab = set((INDEX_DIR / "checkpoints" / "pinyin.vocab").read_text(encoding="utf-8").split())
    return [f"{word}: {say}" for word, say in lex.items() for m in MARK.finditer(say)
            if any(py.upper() not in vocab for py in m[2].split())]


# ---------- 新旧对比（生产 03 号票） ----------

def count(text: str, target: list[str]) -> int:
    """text 的拼音里 target 这串拼音出现几次（不重叠）。"""
    s, n, i = syllables(text), 0, 0
    while i + len(target) <= len(s):
        if s[i:i + len(target)] == target:
            n, i = n + 1, i + len(target)
        else:
            i += 1
    return n


def compare(old: str, new: str, names: list[str], cands: str | None = None, word: str = "雪菜"):
    """两版送进 TTS 的文本不一样的句子（读音表、句首「～」改过的），挑中的成品每句连放两遍：旧 → 新，拼成一个 MP3，
    附时间表和两遍的识别结果，放 trials/pronunciation/。
    另把含 word 的句子 0–3 号候选都识别一遍，数 word 读对了几处：识别结果的拼音里有 word 的拼音（按读音表注音）
    就算对，所以同音字算对。新版缺的候选先按同样的种子补上；旧版的候选取 cands 版（默认同 old），要每句 4 个都在。"""
    import numpy as np
    import soundfile as sf

    from . import index_passage as ip
    from .asrcheck import TorchWhisper
    from .mix import db, process, to_mp3
    from .passage import load_voices, stamp
    from .scriptize import load_cast
    from .synth import load_yaml, tts_text
    from .trial import BOOK
    from .voices import SR

    cast, voices, lex = load_cast(BOOK / "cast.yaml"), load_voices(), load_yaml(BOOK / "lexicon.yaml")
    model, _, variant = new.partition("-")
    anchors = ip.clipped_anchors(variant, voices)
    cands = cands or old
    target = syllables(tts_text({"text": word, "type": "旁白"}, lex))
    sent = {}  # 旧版每句送进 TTS 的文本：从它按轮生成的任务里取
    for f in sorted((ip.WORK / old / "轮次").glob("第*轮.json")):
        for j in json.loads(f.read_text(encoding="utf-8"))["jobs"]:
            p = Path(j["out"])
            sent[(p.parent.parent.name, int(p.stem.split("-")[0]))] = j["text"]
    todo, jobs = [], []
    for name in names:
        _, plan = ip.plan(variant, name, cast, voices, anchors, lex)
        raw = ip.WORK / new / name / "raw"
        for i, s in plan.items():
            if sent.get((name, i), s["text"]) == s["text"]:  # 没变，或者旧版没合成这句
                continue
            todo.append((name, s, sent[(name, i)]))
            if word in plain(s["text"]):
                jobs += [ip.job(raw, s, k) for k in range(ip.TAKES) if not ip.take_path(raw, s, k).exists()]
    if jobs:
        ip.synthesize(model, jobs, ip.WORK / new / "轮次" / "对比补候选.json")

    whisper = TorchWhisper()

    def hear(path: Path) -> str:
        a, sr = sf.read(path, dtype="float32")
        return whisper.transcribe_audio(a, sr)

    gap_in, gap_out = 0.7, 2.0
    chunks, pos, table, words, recs = [], 0.0, [], [], []
    for name, s, before in todo:
        r, i, seg = s["row"], s["row"]["idx"], name.split(" ")[-1]
        q = {v: json.loads((ip.WORK / v / name / "qc.json").read_text(encoding="utf-8"))[str(i)] for v in (old, new)}
        pan = cast["characters"].get(r["speaker"], {}).get("pan", 0.0)
        starts, heard = [], []
        for k, v in enumerate((old, new)):
            f = ip.WORK / v / name / "wav" / f"{i:04d}.wav"
            x = sf.read(f, dtype="float32")[0]
            st = db(process(x, r, pan, True), q[v].get("gain", 0.0)).astype(np.float32)
            g = gap_in if k == 0 else gap_out
            starts.append(stamp(pos))
            chunks += [st, np.zeros((int(g * SR), 2), dtype=np.float32)]
            pos += len(st) / SR + g
            heard.append(hear(f))
            recs.append({"version": v, "name": name, "idx": i, "take": q[v]["take"], "text": (before, s["text"])[k],
                         "hyp": heard[-1], "misread": round(misread((before, s["text"])[k], heard[-1], lex), 3)})
        text = r["text"].replace("\n", " ").replace("|", "｜")
        table.append(f"| {starts[0]} | {starts[1]} | {seg} | {i} | {r['speaker']} | {text} "
                     f"| {q[old]['take']} → {q[new]['take']} | {heard[0]} | {heard[1]} |")
        if word not in plain(s["text"]):
            continue
        n = plain(s["text"]).count(word)
        marks = {}
        for v, ref in ((cands, before), (new, s["text"])):
            marks[v] = []
            for k in range(ip.TAKES):
                hyp = hear(ip.WORK / v / name / "raw" / f"{i:04d}-{k}.wav")
                ok = min(count(hyp, target), n)
                marks[v].append((ok, hyp))
                recs.append({"version": v, "name": name, "idx": i, "cand": k, "text": ref, "hyp": hyp, "word": n,
                             "word_ok": ok})
        words.append((seg, i, r["speaker"], text, n, marks))

    out = np.concatenate(chunks)
    out *= min(1.0, 0.97 / float(np.abs(out).max()))
    trial = BOOK / "trials" / "pronunciation"
    trial.mkdir(parents=True, exist_ok=True)
    mp3 = trial / "新旧对比.mp3"
    to_mp3(out, mp3, {"album": "白色相簿2 剧本全集（试做）", "title": "读音-新旧对比", "artist": "AI 有声书试做"})
    with (trial / "识别.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in recs)

    def tally(v: str) -> str:
        ok = sum(m[v][k][0] for *_, m in words for k in range(ip.TAKES))
        n = sum(w[4] for w in words) * ip.TAKES
        return f"{ok}/{n}（{ok / n:.0%}）"

    md = ["# 读音新旧对比", "",
          f"送进 TTS 的文本变了的句子（读音表加了「{word}」的拼音标注，句首的「～」去掉），挑中的成品每句连放两遍："
          f"旧版（`{old}`）→ 新版（`{new}`）。同一句两遍隔 {gap_in:g} 秒，换句隔 {gap_out:g} 秒，混音同正式流程。"
          "候选是第几号（种子 = 句号 × 100 + 候选号）。识别用 Whisper large-v3-turbo，从干声识别。", "",
          "| 旧 | 新 | 片段 | # | 谁 | 台词 | 候选 | 旧版识别 | 新版识别 |", "|---|---|---|---|---|---|---|---|---|",
          *table, "",
          f"## 「{word}」读对的比例", "",
          f"含「{word}」的 {len(words)} 句、{sum(w[4] for w in words)} 处，每句 0–3 号候选（同样的种子）都识别一遍："
          f"识别结果的拼音里有 {' '.join(target)} 就算这一处读对（同音字也算对）。旧版的候选取 `{cands}`，新版 `{new}`"
          "（按轮生成没用到的候选也补上了）。识别有语言模型兜底，读得不准也可能写成「" + word + "」，以试听为准。", "",
          f"- 旧版：{tally(cands)}", f"- 新版：{tally(new)}", "",
          "| 片段 | # | 谁 | 台词 | 处 | 旧 0–3 号 | 新 0–3 号 | 没识别出来的 |", "|---|---|---|---|---|---|---|---|"]
    for seg, i, who, text, n, marks in words:
        cells = ["".join("✓" if ok == n else "✗" for ok, _ in marks[v]) for v in (cands, new)]
        miss = "；".join(f"{'旧' if v == cands else '新'} {k} 号「{hyp}」" for v in (cands, new)
                        for k, (ok, hyp) in enumerate(marks[v]) if ok < n)
        md.append(f"| {seg} | {i} | {who} | {text} | {n} | {cells[0]} | {cells[1]} | {miss} |")
    (trial / "新旧对比.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print(mp3, f"{len(out) / SR / 60:.1f} 分钟，{len(table)} 句；{word}：旧 {tally(cands)}，新 {tally(new)}")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    if cmd == "检查":
        from .synth import load_yaml
        from .trial import BOOK

        bad = unknown(load_yaml(BOOK / "lexicon.yaml"))
        print("\n".join(bad) if bad else "读音表的拼音标注都在 IndexTTS 的拼音表里")
    elif cmd == "对比":
        cands = None
        if "--旧候选" in args:
            i = args.index("--旧候选")
            cands, args = args[i + 1], args[:i] + args[i + 2:]
        compare(args[0], args[1], args[2:], cands)
