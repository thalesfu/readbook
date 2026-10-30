"""逐句表演设置（11 号票）：拿性格卡、整段场景和标好的剧本，让 LLM 给每句定更细的表演，交给 IndexTTS-2.5。

已有剧本的字段一个不动，表演设置另存一份（books/wa2/trials/perform/表演-<剧本>.jsonl），这样和基线能同一句、
同一种子配对。一次 `claude -p` 给齐三种做法要的数，三种做法只差"情绪从哪来"：
- A（2.5-表演A）：向量照旧按 mood 查 index_passage.EMOTION，乘上 strength（emo_alpha）；
- B（2.5-表演B）：LLM 直接给 8 维向量 vec，和查表的一样过 normalize_emo_vec；
- C（2.5-表演C）：LLM 写一句情感描述 emo，交给 IndexTTS 自带的情感模型（QwenEmotion）转成向量，
  再过 normalize_emo_vec（indextts_batch）。
三种都用同一套 tempo 和 gain：
- tempo（语速倍数）进合成：duration_factor = 1/tempo，挑选时的目标语速也乘 tempo，免得 finish 又拉回去；
- gain（dB）进混音：每句先拉到同一响度，再加减（mix.assemble）。
用户试听定了 B，时长放混音：2.5-表演B混音 版情绪同 B，合成不拉长，候选读进来先按 tempo 用 atempo 变速，再照常挑选。
讲述模式（叙述者情绪平常的旁白和心里话，拿讲述定妆音当情感参考）不换情绪，只用 tempo 和 gain。

    uv run python -m readbook.perform 标注 小样|"2010-11-25 〈2002〉"   # 调 claude -p，写表演设置
    uv run python -m readbook.index_passage 2.5-表演A 小样              # 合成、挑选、混音（B、C、B混音 同）
    uv run python -m readbook.perform 时长 小样|"2010-11-25 〈2002〉"   # 「呼」和轻声：合成拉长 vs 混音变速
    uv run python -m readbook.perform 合集                              # 试听合集：基线 → A → B
"""

import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import qc
from .bakeoff import SEG
from .moods import NARRATION
from .scriptize import ask_claude, load_cast
from .synth import load_yaml, tts_text
from .trial import BOOK, read_jsonl, write_jsonl

PERF = BOOK / "trials" / "perform"
BASELINE = {"小样": "2.5-原声平均锚点新规", "2010-11-25 〈2002〉": "2.5-原声平均锚点全员"}  # 19 号票成品的候选
VERSIONS = {"表演A": "A", "表演B": "B", "表演C": "C", "表演B混音": "B"}
MIX = {"表演B混音"}  # 语速放混音（用户试听时长 A/B 定的）：合成不拉长，挑选前每个候选先用 atempo 变速
DIMS = ["喜", "怒", "哀", "惧", "厌恶", "忧郁", "惊讶", "平静"]
TEMPO = (0.5, 1.25)
GAIN = (-9.0, 3.0)


def script_of(name: str) -> str:
    """片段 → 剧本文件名：小样是 2011-01-01 那份剧本里挑的句子，表演设置按整份剧本标。"""
    return SEG if name == "小样" else name


SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer"},
                    "strength": {"type": "number"},
                    "vec": {"type": "array", "items": {"type": "number"}, "minItems": 8, "maxItems": 8},
                    "emo": {"type": "string"},
                    "tempo": {"type": "number"},
                    "gain": {"type": "number"},
                    "why": {"type": "string"},
                },
                "required": ["idx", "strength", "vec", "emo", "tempo", "gain", "why"],
            },
        }
    },
    "required": ["lines"],
}

RULES = """\
你在为一部中文小说制作多角色有声书。剧本已经逐行标好：类别（type）、说话人（speaker）、声音效果（effect）、
情绪（mood）、表演提示（instruct）、这一行前面要留的停顿秒数（pause）。现在要给每一行定更细的表演设置，
交给语音合成引擎 IndexTTS-2.5。这部小说是第一人称，叙述者是「{narrator}」。

## 引擎能做什么
- 音色固定用每个角色自己的参考录音。情绪靠 8 维情感向量：[{dims}]。引擎把这 8 种情绪的示例声音按权重混进去，
  剩下的部分用角色本人平常的声音。权重越大情绪越浓，但也越容易把音色带偏、不像本人（总和会被压到 0.8 以内）。宁淡勿浓。
- 现在的做法是按 mood 查一张固定表（「平常」不加向量，就是本人平常的声音）：
{table}
- 标了 "mode": "讲述" 的行是叙述者情绪平常的旁白和心理活动，走讲述模式（拿他自己一段讲述录音定情绪，不用向量）。
  这些行只看 tempo 和 gain，strength 填 1、vec 全填 0、emo 留空。
- 只有标点、没有字的行（如「………」）不合成，也按默认值填。

## 每行要给的
- strength：0–1，查表那个向量用几成（1 = 表里的原值）。mood 是「平常」的行填 1。
- vec：8 个 0–1 的数，直接给这一行的情感向量，顺序同上。可以拿上表当起点，按人物和处境调；情绪可以混合，
  比如强撑出来的笑意 = 喜 + 忧郁。不需要情绪、就用本人平常声音的行全填 0。
- emo：一句不超过 20 字的中文情感描述，交给引擎自带的情感模型转成向量。它只认上面 8 种情绪和浓淡，
  写成「什么情绪、有多浓」，比如「有点难过，又强撑着笑」「压着火，冷冷的」。不需要情绪的行留空。
- tempo：语速倍数，{tempo_lo}–{tempo_hi}，1 = 这个角色平常的语速，整句一起变慢变快。轻声细语、迟疑、叹气要慢；
  着急、慌张可以快。只有一两个字的叹词（「呼」「啊」「嗯」）如果是长长地叹气、拖长了的应声，用 0.5–0.7 把它拉长。
- gain：音量，单位分贝，{gain_lo:g} 到 +{gain_hi:g}，0 = 平常的响度。轻声、自言自语、耳语要轻（−3 到 −9）；
  喊叫、兴奋可以稍响（最多 +{gain_hi:g}）。
- why：不超过 25 字，说依据：人物性格、处境、前后文。

## 怎么定
- 先读人物卡，再通读整段，弄清每一行在场景里是什么处境、前后发生了什么，再逐行定。
- 同一种 mood，人物和处境不同，设置就可以不同：习惯压抑情绪的人难过时更淡、更低、更慢；习惯强撑笑脸的人
  难过时带一点喜；慌乱时快，试探时慢而轻。
- 已有的 mood 和 instruct 是参考，不要和它们差得太远；拿不准就接近默认值（strength 1、vec 照表、tempo 1、gain 0）。
- 声音效果（电话、回忆）在混音时另加，这里不用管。

## 人物卡
{cards}

## 输出
按 schema 返回 lines，idx 与输入一一对应，一行不漏。
"""


def cards_text(cast: dict, names: list[str]) -> str:
    """出场人物的性格卡；没有卡的用 desc。"""
    out = []
    for n in names:
        c = cast["characters"][n]
        if "card" in c:
            out.append(f"### {n}\n" + "\n".join(f"- {k}：{v}" for k, v in c["card"].items()))
        else:
            out.append(f"### {n}\n- {c['desc']}")
    return "\n".join(out)


def narration(row: dict, cast: dict) -> bool:
    """走讲述模式的行（同 index_passage.plan_row）。"""
    return row["speaker"] == cast["narrator"] and row["type"] in ("旁白", "心理活动") \
        and row.get("mood", "平常") == "平常"


def build_prompt(rows: list[dict], cast: dict) -> str:
    from .index_passage import EMOTION

    table = "\n".join(f"  - {m}：" + ("不加" if v is None else "、".join(f"{d} {x:g}" for d, x in zip(DIMS, v) if x))
                      for m, v in EMOTION.items() if m in ("平常", "开心", "兴奋", "生气", "难过", "哭腔", "害羞",
                                                         "冷淡", "惊讶", "轻声"))
    names = list(dict.fromkeys(r["speaker"] for r in rows))
    head = RULES.format(narrator=cast["narrator"], dims="、".join(DIMS), table=table, cards=cards_text(cast, names),
                        tempo_lo=TEMPO[0], tempo_hi=TEMPO[1], gain_lo=GAIN[0], gain_hi=GAIN[1])
    body = []
    for r in rows:
        x = {k: r[k] for k in ("idx", "type", "speaker", "effect", "mood", "instruct", "text")}
        if r.get("pause"):
            x["pause"] = r["pause"]
        if narration(r, cast):
            x["mode"] = NARRATION
        body.append(json.dumps(x, ensure_ascii=False))
    return f"{head}\n## 输入\n" + "\n".join(body)


def clean(m: dict) -> dict:
    clip = lambda x, lo, hi: min(max(float(x), lo), hi)  # noqa: E731
    return {"idx": m["idx"], "strength": round(clip(m["strength"], 0, 1), 2),
            "vec": [round(clip(x, 0, 1), 2) for x in m["vec"]], "emo": m["emo"].strip(),
            "tempo": round(clip(m["tempo"], *TEMPO), 2), "gain": round(clip(m["gain"], *GAIN), 1), "why": m["why"]}


def label(name: str, model: str | None = None):
    """整份剧本交给 claude -p 标表演设置，写 表演-<剧本>.jsonl；用量追加到 表演-成本.jsonl。"""
    script = script_of(name)
    rows = read_jsonl(BOOK / "script" / f"{script}.jsonl")
    cast = load_cast(BOOK / "cast.yaml")
    prompt = build_prompt(rows, cast)
    meta = {}
    data = ask_claude(prompt, model, SCHEMA, meta)
    got = {m["idx"]: clean(m) for m in data["lines"]}
    missing = [r["idx"] for r in rows if r["idx"] not in got]
    if missing:
        raise ValueError(f"LLM 漏了 {len(missing)} 行：{missing[:10]}")
    PERF.mkdir(parents=True, exist_ok=True)
    write_jsonl(PERF / f"表演-{script}.jsonl", [got[r["idx"]] for r in rows])
    with (PERF / "表演-成本.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"script": script, "lines": len(rows), "prompt_chars": len(prompt),
                            "output_chars": len(json.dumps(data["lines"], ensure_ascii=False)), **meta},
                           ensure_ascii=False) + "\n")
    print(PERF / f"表演-{script}.jsonl", meta.get("total_cost_usd"), meta.get("duration_ms"))


def load(name: str) -> dict[int, dict]:
    return {m["idx"]: m for m in read_jsonl(PERF / f"表演-{script_of(name)}.jsonl")}


def apply(p: dict, m: dict, ver: str, mix: bool = False) -> dict:
    """按做法 A/B/C 改一句的合成设置（index_passage.plan_row 的结果）：情绪、强度、目标语速，以及在哪拉长：
    合成时（df = duration_factor），或 mix=True 时合成不拉长、候选读进来先按 tempo 变速（index_passage.pick）。
    slow 是这句放慢的倍数，质检的时长上限跟着放宽。"""
    p = {**p, "rate": p["rate"] * m["tempo"], "df": 1.0 if mix else round(1 / m["tempo"], 3),
         "slow": round(1 / m["tempo"], 3)}
    if mix:
        p["tempo"] = m["tempo"]
    if p["how"] == NARRATION:
        return p
    if ver == "A":
        if p.get("emo_vector"):
            p["alpha"] = m["strength"]
    elif ver == "B":
        p["emo_vector"] = m["vec"] if sum(m["vec"]) > 0 else None
    elif ver == "C":
        p["emo_vector"], p["emo_text"] = None, m["emo"] or None
    return p


def note(m: dict, p: dict, ver: str, emo_vecs: dict | None = None) -> str:
    """对照剧本里的「表演」一格。"""
    bits = [f"语速×{m['tempo']:g}", f"音量{m['gain']:+g}dB"]
    if p["how"] == NARRATION:
        bits.append("讲述")
    elif ver == "A" and p.get("emo_vector"):
        bits.append(f"强度{m['strength']:g}")
    elif ver == "B" and p.get("emo_vector"):
        bits.append(vec_text(p["emo_vector"]))
    elif ver == "C" and p.get("emo_text"):
        v = (emo_vecs or {}).get(p["emo_text"])
        bits.append(f"「{p['emo_text']}」" + (f"→{vec_text(v)}" if v else ""))
    return "，".join(bits) + f"（{m['why']}）"


def vec_text(v: list[float]) -> str:
    return " ".join(f"{d}{x:.2f}".replace("0.", ".") for d, x in zip(DIMS, v) if x >= 0.005) or "无"


# ---------- 和基线一样的任务不重新合成 ----------

@functools.cache
def digest(path: str) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def job_key(j: dict) -> tuple:
    return (j["text"], digest(j["spk"]), tuple(j.get("emo_vector") or ()), j.get("emo_audio") and digest(j["emo_audio"]),
            j["emo_alpha"], j["seed"], j.get("duration_factor", 1.0), j.get("emo_text"))


def sources(variant: str, name: str) -> list[str]:
    """哪些版本的候选可以直接拿来用：基线；B混音版还有 B 版里本来就没拉长的。"""
    return [BASELINE[name]] + (["2.5-表演B"] if variant == "表演B混音" else [])


def reuse(jobs: list[dict], base: Path, raw: Path) -> int:
    """和基线完全一样的任务（同一句、同一种子、同样的说话人参考和情绪、不拉长）直接拷基线的候选，耗时照抄基线。
    说话人参考按文件内容比：两段基线的锚点在不同目录，春希、雪菜、依绪的内容相同。"""
    spec = json.loads((base / "jobs.json").read_text(encoding="utf-8"))["jobs"]
    timing = {Path(t["out"]).name: t for t in map(json.loads, (base / "timing.jsonl").read_text(encoding="utf-8")
                                                  .splitlines()) if "out" in t}
    have = {job_key(j): j for j in spec if Path(j["out"]).exists()}
    n = 0
    with (raw / "timing.jsonl").open("a", encoding="utf-8") as log:
        for j in jobs:
            b, dst = have.get(job_key(j)), Path(j["out"])
            if b and not dst.exists():
                shutil.copyfile(b["out"], dst)
                log.write(json.dumps({**timing[Path(b["out"]).name], "out": j["out"], "reused": b["out"]},
                                     ensure_ascii=False) + "\n")
                n += 1
    print(f"拿 {base.parent.parent.name} 现成的候选 {n}/{len(jobs)} 个", flush=True)
    return n


# ---------- 时长：合成拉长 vs 混音变速 ----------

def duration_ab(name: str):
    """「呼」这类一两个字的叹词和轻声句：同一个候选（A 版挑中的那个种子），一份在合成时拉长（duration_factor，即 A 版成品），
    一份不拉长、合成后用 atempo 变速，再和基线成品排在一起出一个短 MP3 和对照表。音量都按 A 版的 gain。"""
    from . import index_passage as ip
    from .bakeoff import finish
    from .mix import SR, db, process, to_mp3
    from .passage import load_rows, load_voices, stamp
    import soundfile as sf

    version, ab = "2.5-表演A", "2.5-表演A-混音变速"
    rows = load_rows(name, None, None, None)
    cast = load_cast(BOOK / "cast.yaml")
    voices = load_voices()
    lex = load_yaml(BOOK / "lexicon.yaml")
    marks = load(name)
    anchors = ip.clipped_anchors("表演A", voices)
    src = ip.WORK / version / name
    qcj = json.loads((src / "qc.json").read_text(encoding="utf-8"))
    spec = {Path(j["out"]).name: j for j in json.loads((src / "raw" / "jobs.json").read_text(encoding="utf-8"))["jobs"]}
    raw = ip.WORK / ab / name / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    picked, jobs = [], []
    for r in rows:
        m, text = marks[r["idx"]], tts_text(r, lex)
        if str(r["idx"]) not in qcj or abs(m["tempo"] - 1) < 0.02:
            continue
        if r.get("mood") != "轻声" and len(qc.SPOKEN.findall(text)) > 2:
            continue
        k = qcj[str(r["idx"])]["take"]
        j = {**spec[f"{r['idx']:04d}-{k}.wav"], "out": str((raw / f"{r['idx']:04d}-{k}.wav").resolve())}
        j.pop("duration_factor", None)
        picked.append((r, k))
        jobs.append(j)
    (raw / "jobs.json").write_text(json.dumps({"version": "2.5", "jobs": jobs}, ensure_ascii=False, indent=0),
                                   encoding="utf-8")
    reuse(jobs, ip.WORK / BASELINE[name] / name / "raw", raw)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1"}
    subprocess.run(["uv", "run", "--project", str(ip.INDEX_DIR), "python", str(ip.BATCH), str(raw / "jobs.json")],
                   cwd=ip.INDEX_DIR, env=env, check=True)

    base_wav = ip.WORK / f"{BASELINE[name]}-重采样" / name / "wav"
    chunks, pos, table = [], 0.0, []
    sil = lambda s: np.zeros((int(s * SR), 2), dtype=np.float32)  # noqa: E731
    for r, k in picked:
        m = marks[r["idx"]]
        p = apply(ip.plan_row(r, cast, voices, anchors), m, "A")
        a = qc.stretch(ip.read_take(raw / f"{r['idx']:04d}-{k}.wav"), m["tempo"])
        a, _ = finish([a], tts_text(r, lex), p)
        pan = cast["characters"].get(r["speaker"], {}).get("pan", 0.0)
        takes = [("基线", sf.read(base_wav / f"{r['idx']:04d}.wav", dtype="float32")[0], 0.0),
                 ("合成拉长", sf.read(src / "wav" / f"{r['idx']:04d}.wav", dtype="float32")[0], m["gain"]),
                 ("混音变速", a, m["gain"])]
        for i, (what, x, gain) in enumerate(takes):
            table.append(f"| {stamp(pos)} | {r['idx']} | {r['speaker']} | {r.get('mood', '')} | {what} "
                         f"| {len(x) / SR:.2f} 秒 | {r['text']} |")
            st = db(process(x, r, pan, True), gain).astype(np.float32)
            chunks += [st, sil(0.8 if i < 2 else 2.0)]
            pos += len(st) / SR + (0.8 if i < 2 else 2.0)
    out = np.concatenate(chunks)
    out *= min(1.0, 0.97 / float(np.abs(out).max()))
    PERF.mkdir(parents=True, exist_ok=True)
    mp3 = PERF / f"{name}-时长AB.mp3"
    to_mp3(out, mp3, {"album": "白色相簿2 剧本全集（试做）", "title": f"{name}-时长AB", "artist": "AI 有声书试做"})
    head = [f"# {name}：时长 A/B（合成拉长 vs 混音变速）", "",
            "每句连放三遍：基线（19 号票成品）、合成拉长（A 版成品，IndexTTS 的 duration_factor = 1/语速倍数）、"
            "混音变速（同一个种子不拉长，合成后用 atempo 变速，再按同样的规则把语速拉回目标）。后两遍音量都按 A 版的 gain。"
            "时长是去掉首尾静音、加混音效果之前的秒数。", "",
            "| 时间 | # | 谁 | 情绪 | 哪一遍 | 时长 | 台词 |", "|---|---|---|---|---|---|---|"]
    (PERF / f"{name}-时长AB.md").write_text("\n".join(head + table) + "\n", encoding="utf-8", newline="\n")
    print(mp3)


# ---------- 试听合集 ----------

LISTEN = {  # 片段 → [(句号, 听什么)]
    "2010-11-25 〈2002〉": [(24, "「呼」够不够长"), (21, "轻声叹词"), (10, "轻声够不够轻、慢"), (12, "轻声够不够轻、慢"),
                          (16, "轻声够不够轻、慢"), (17, "轻声够不够轻、慢"), (5, "雪菜强撑笑脸"), (6, "雪菜强撑笑脸"),
                          (11, "雪菜强撑笑脸"), (19, "雪菜强撑笑脸"), (61, "春希轻唤（音高）"), (62, "春希压抑的心里话"),
                          (63, "春希压抑的心里话")],
    "小样": [(0, "开头的轻叹"), (3, "武也平常台词加了情绪（音色变没变）"), (4, "春希无奈"), (59, "雪菜戴着笑脸"),
           (65, "雪菜轻声"), (67, "雪菜轻声"), (72, "春希轻声"), (79, "春希轻声"), (81, "春希轻声"), (83, "春希轻声"),
           (78, "雪菜哭腔"), (84, "雪菜哭腔")],
}


def sampler():
    """要听的句子每句连放三遍：基线（19 号票成品）→ A → B，两段拼成一个 MP3 和一张表。混音同正式流程，
    A、B 按各自的 gain 加减音量。"""
    from . import index_passage as ip
    from .mix import SR, db, process, to_mp3
    from .passage import load_rows, stamp
    import soundfile as sf

    cast = load_cast(BOOK / "cast.yaml")
    gap_in, gap_out = 0.7, 2.0
    chunks, pos, table = [], 0.0, []
    for name, items in LISTEN.items():
        rows = {r["idx"]: r for r in load_rows(name, None, None, None)}
        marks = load(name)
        for i, what in items:
            r, m = rows[i], marks[i]
            pan = cast["characters"].get(r["speaker"], {}).get("pan", 0.0)
            starts = []
            for k, (d, gain) in enumerate(((ip.WORK / f"{BASELINE[name]}-重采样" / name, 0.0),
                                           (ip.WORK / "2.5-表演A" / name, m["gain"]),
                                           (ip.WORK / "2.5-表演B" / name, m["gain"]))):
                x = sf.read(d / "wav" / f"{i:04d}.wav", dtype="float32")[0]
                st = db(process(x, r, pan, True), gain).astype(np.float32)
                g = gap_in if k < 2 else gap_out
                starts.append(stamp(pos))
                chunks += [st, np.zeros((int(g * SR), 2), dtype=np.float32)]
                pos += len(st) / SR + g
            b = vec_text(m["vec"]) if sum(m["vec"]) else "不加"
            table.append(f"| {' | '.join(starts)} | {'小样' if name == '小样' else '〈2002〉'} | {i} | {r['speaker']} "
                         f"| {r.get('mood', '')} | {r['text']} | {what} | 语速×{m['tempo']:g}，音量{m['gain']:+g}dB；"
                         f"A 强度{m['strength']:g}；B {b}（{m['why']}） |")
    out = np.concatenate(chunks)
    out *= min(1.0, 0.97 / float(np.abs(out).max()))
    mp3 = PERF / "表演-试听合集.mp3"
    to_mp3(out, mp3, {"album": "白色相簿2 剧本全集（试做）", "title": "表演-试听合集", "artist": "AI 有声书试做"})
    head = ["# 表演设置试听合集（基线 → A → B）", "",
            f"每句连放三遍，顺序固定：基线（19 号票成品）、A（查表向量 × 强度）、B（Claude 直接给向量）。"
            f"同一句三遍之间隔 {gap_in:g} 秒，换句隔 {gap_out:g} 秒。A、B 的语速和音量一样，只差情绪。", "",
            "| 基线 | A | B | 片段 | # | 谁 | 情绪 | 台词 | 听什么 | 表演设置 |", "|---|---|---|---|---|---|---|---|---|---|"]
    (PERF / "表演-试听合集.md").write_text("\n".join(head + table) + "\n", encoding="utf-8", newline="\n")
    print(mp3, f"{len(out) / SR / 60:.1f} 分钟")


if __name__ == "__main__":
    if sys.argv[1] == "标注":
        label(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif sys.argv[1] == "时长":
        duration_ab(sys.argv[2])
    elif sys.argv[1] == "合集":
        sampler()
