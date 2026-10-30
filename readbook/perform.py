"""逐句表演设置：拿性格卡、整段场景和标好的剧本，让 LLM 给每句定更细的表演，交给 IndexTTS-2.5。

正式流程（合成规格第三节，生产 02 号票）：剧本化、停顿之后再调一次 `claude -p`（label），每行加
perform: {vec, tempo, gain}，已有字段一个不动（with_perform）。新剧本在 passage.load_rows 里接上；已有的剧本
先「标注」到 books/wa2/trials/perform/表演-<剧本>.jsonl 看，认可了再「写回」剧本。定版 2.5-表演B混音 从剧本行读：
- vec：8 维情感向量，合成时代替按 mood 查表，照旧过 normalize_emo_vec；全 0 就是本人平常的声音；
- tempo：语速倍数，放混音：合成不拉长，候选读进来先用 atempo 变速，挑选的目标语速也乘它；
- gain（dB）：每句先拉到同一响度，过完后期链再加减（mix.assemble）。
讲述模式（叙述者情绪平常的旁白和心里话，拿讲述定妆音当情感参考）不换情绪，vec 填 0，只用 tempo 和 gain。
没有 perform 的行照旧：情绪按 mood 查表，语速 1，音量 0。

11 号票试做时一次给齐三种做法要的数（另存在 表演-<剧本>.jsonl，还有 strength、emo、why），三种只差"情绪从哪来"，
A/B/C 三个试做版照旧读那一份：
- A（2.5-表演A）：向量照旧按 mood 查 index_passage.EMOTION，乘上 strength（emo_alpha）；
- B（2.5-表演B）：LLM 直接给 8 维向量 vec；
- C（2.5-表演C）：LLM 写一句情感描述 emo，交给 IndexTTS 自带的情感模型（QwenEmotion）转成向量，
  再过 normalize_emo_vec（indextts_batch）；
- 三种的 tempo 都进合成：duration_factor = 1/tempo，挑选时的目标语速也乘 tempo，免得 finish 又拉回去。
用户试听定了 B，时长放混音，就是 2.5-表演B混音。

    uv run python -m readbook.perform 标注 小样|"2010-11-25 〈2002〉"   # 调 claude -p，写 表演-<剧本>.jsonl
    uv run python -m readbook.perform 写回 "2010-11-25 〈2002〉" …      # 认可后把 vec、tempo、gain 写进剧本
    uv run python -m readbook.perform why 小样                          # why 留不留：两种各标两遍，比出入
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
# 语速放混音（用户试听时长 A/B 定的）：合成不拉长，挑选前每个候选先用 atempo 变速。定版，表演设置从剧本行读
MIX = {"表演B混音"}
DIMS = ["喜", "怒", "哀", "惧", "厌恶", "忧郁", "惊讶", "平静"]
TEMPO = (0.5, 1.25)
GAIN = (-9.0, 3.0)
FIELDS = ("vec", "tempo", "gain")  # 剧本行的 perform 只存这三个
MODEL = "claude-opus-5-5"  # 11 号票验证时用的；换模型要重新试听
WHY = False  # 不让 LLM 每行再写一句依据：生产 02 号票比过，留不留差得不比标两遍多，去掉快约 20%（用户定的）


def script_of(name: str) -> str:
    """片段 → 剧本文件名：小样是 2011-01-01 那份剧本里挑的句子，表演设置按整份剧本标。"""
    return SEG if name == "小样" else name


def schema(why: bool = WHY) -> dict:
    line = {"idx": {"type": "integer"},
            "vec": {"type": "array", "items": {"type": "number"}, "minItems": 8, "maxItems": 8},
            "tempo": {"type": "number"}, "gain": {"type": "number"}}
    if why:
        line["why"] = {"type": "string"}
    return {"type": "object", "required": ["lines"],
            "properties": {"lines": {"type": "array",
                                     "items": {"type": "object", "properties": line, "required": list(line)}}}}

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
  这些行只看 tempo 和 gain，vec 全填 0。
- 只有标点、没有字的行（如「………」）不合成，也按默认值填。

## 每行要给的
- vec：8 个 0–1 的数，直接给这一行的情感向量，顺序同上。可以拿上表当起点，按人物和处境调；情绪可以混合，
  比如强撑出来的笑意 = 喜 + 忧郁。不需要情绪、就用本人平常声音的行全填 0。
- tempo：语速倍数，{tempo_lo}–{tempo_hi}，1 = 这个角色平常的语速，整句一起变慢变快。轻声细语、迟疑、叹气要慢；
  着急、慌张可以快。只有一两个字的叹词（「呼」「啊」「嗯」）如果是长长地叹气、拖长了的应声，用 0.5–0.7 把它拉长。
- gain：音量，单位分贝，{gain_lo:g} 到 +{gain_hi:g}，0 = 平常的响度。轻声、自言自语、耳语要轻（−3 到 −9）；
  喊叫、兴奋可以稍响（最多 +{gain_hi:g}）。
{why}
## 怎么定
- 先读人物卡，再通读整段，弄清每一行在场景里是什么处境、前后发生了什么，再逐行定。
- 同一种 mood，人物和处境不同，设置就可以不同：习惯压抑情绪的人难过时更淡、更低、更慢；习惯强撑笑脸的人
  难过时带一点喜；慌乱时快，试探时慢而轻。
- 已有的 mood 和 instruct 是参考，不要和它们差得太远；拿不准就接近默认值（vec 照表、tempo 1、gain 0）。
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


def build_prompt(rows: list[dict], cast: dict, why: bool = WHY) -> str:
    from .index_passage import EMOTION

    table = "\n".join(f"  - {m}：" + ("不加" if v is None else "、".join(f"{d} {x:g}" for d, x in zip(DIMS, v) if x))
                      for m, v in EMOTION.items() if m in ("平常", "开心", "兴奋", "生气", "难过", "哭腔", "害羞",
                                                         "冷淡", "惊讶", "轻声"))
    names = list(dict.fromkeys(r["speaker"] for r in rows))
    head = RULES.format(narrator=cast["narrator"], dims="、".join(DIMS), table=table, cards=cards_text(cast, names),
                        tempo_lo=TEMPO[0], tempo_hi=TEMPO[1], gain_lo=GAIN[0], gain_hi=GAIN[1],
                        why="- why：不超过 25 字，说依据：人物性格、处境、前后文。\n" if why else "")
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
    out = {"idx": m["idx"], "vec": [round(clip(x, 0, 1), 2) for x in m["vec"]],
           "tempo": round(clip(m["tempo"], *TEMPO), 2), "gain": round(clip(m["gain"], *GAIN), 1)}
    if "why" in m:
        out["why"] = m["why"]
    return out


def label(rows: list[dict], cast: dict, model: str | None = MODEL, why: bool = WHY,
          meta: dict | None = None) -> dict[int, dict]:
    """标好停顿的剧本整份交给 claude -p → {idx: {idx, vec, tempo, gain[, why]}}，一行不漏。
    meta 给了就填进这次调用的用量和提示、返回的字数，算成本用。"""
    prompt = build_prompt(rows, cast, why)
    data = ask_claude(prompt, model, schema(why), meta)
    got = {m["idx"]: clean(m) for m in data["lines"]}
    missing = [r["idx"] for r in rows if r["idx"] not in got]
    if missing:
        raise ValueError(f"LLM 漏了 {len(missing)} 行：{missing[:10]}")
    if meta is not None:
        meta.update(prompt_chars=len(prompt), output_chars=len(json.dumps(data["lines"], ensure_ascii=False)))
    return {r["idx"]: got[r["idx"]] for r in rows}


def with_perform(rows: list[dict], marks: dict[int, dict], cast: dict) -> list[dict]:
    """给每行补上 perform：只取 vec、tempo、gain，讲述模式的行 vec 填 0；其他字段不动，已有的 perform 整个换掉。"""
    return [{**r, "perform": {"vec": [0.0] * len(DIMS) if narration(r, cast) else marks[r["idx"]]["vec"],
                              **{k: marks[r["idx"]][k] for k in FIELDS[1:]}}} for r in rows]


def load(name: str) -> dict[int, dict]:
    """表演-<剧本>.jsonl：标注出来待看的，或 11 号票试做的那一份（A/B/C 版读它）。"""
    return {m["idx"]: m for m in read_jsonl(PERF / f"表演-{script_of(name)}.jsonl")}


def cmd_label(name: str):
    """已有的剧本整份标一遍，写 表演-<剧本>.jsonl 先看，不动剧本；用量追加到 表演-成本.jsonl。"""
    script = script_of(name)
    rows = read_jsonl(BOOK / "script" / f"{script}.jsonl")
    meta = {}
    marks = label(rows, load_cast(BOOK / "cast.yaml"), meta=meta)
    write_jsonl(PERF / f"表演-{script}.jsonl", list(marks.values()))
    with (PERF / "表演-成本.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"script": script, "lines": len(rows), **meta}, ensure_ascii=False) + "\n")
    print(PERF / f"表演-{script}.jsonl", meta.get("total_cost_usd"), meta.get("duration_ms"))


def cmd_write(names: list[str]):
    """认可了的 表演-<剧本>.jsonl 写进剧本（with_perform）。"""
    cast = load_cast(BOOK / "cast.yaml")
    for script in dict.fromkeys(map(script_of, names)):
        path = BOOK / "script" / f"{script}.jsonl"
        rows = with_perform(read_jsonl(path), load(script), cast)
        write_jsonl(path, rows)
        print(path, len(rows), "行")


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
    return "，".join(bits) + (f"（{m['why']}）" if m.get("why") else "")


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


# ---------- why 留不留（生产 02 号票） ----------

WHY_AB = PERF / "why"


def why_ab(name: str, n: int = 2):
    """同一份剧本，留 why 和去 why 各标 n 遍（并行），看两种做法之间差得是不是比同一种做法标两遍还多。
    每遍的标注和用量放 trials/perform/why/，已有的不重标；对照还有 11 号票试做的那一份（原版，七个字段）。"""
    from concurrent.futures import ThreadPoolExecutor
    from itertools import combinations

    script = script_of(name)
    rows = read_jsonl(BOOK / "script" / f"{script}.jsonl")
    cast = load_cast(BOOK / "cast.yaml")
    cost = WHY_AB / "成本.jsonl"

    def one(tag: str) -> dict[int, dict]:
        f = WHY_AB / f"{script}-{tag}.jsonl"
        if not f.exists():
            meta = {}
            write_jsonl(f, list(label(rows, cast, why=tag.startswith("留"), meta=meta).values()))
            with cost.open("a", encoding="utf-8") as log:
                log.write(json.dumps({"script": script, "run": tag, "lines": len(rows), **meta}, ensure_ascii=False) + "\n")
        return {m["idx"]: m for m in read_jsonl(f)}

    tags = [f"{w}why-{i + 1}" for w in ("留", "去") for i in range(n)]
    with ThreadPoolExecutor(len(tags)) as ex:
        runs = dict(zip(tags, ex.map(one, tags)))
    runs["原版"] = load(script)
    spoken = [r for r in rows if qc.SPOKEN.search(r["text"])]
    acted = [r for r in spoken if not narration(r, cast)]  # vec 只在这些行起作用
    l1 = lambda a, b: sum(abs(x - y) for x, y in zip(a, b))  # noqa: E731
    top = lambda v: int(np.argmax(v)) if sum(v) > 0 else None  # noqa: E731

    def dist(a: str, b: str) -> str:
        x, y = runs[a], runs[b]
        i = [r["idx"] for r in acted]
        return (f"| {a} ↔ {b} | {np.mean([l1(x[k]['vec'], y[k]['vec']) for k in i]):.3f} "
                f"| {sum(top(x[k]['vec']) != top(y[k]['vec']) for k in i)}/{len(i)} "
                f"| {np.mean([abs(x[r['idx']]['tempo'] - y[r['idx']]['tempo']) for r in spoken]):.3f} "
                f"| {np.mean([abs(x[r['idx']]['gain'] - y[r['idx']]['gain']) for r in spoken]):.2f} |")

    pairs = list(combinations(tags, 2))
    same = [p for p in pairs if p[0][0] == p[1][0]]
    lines = [f"# 表演设置：why 留不留（{script}）", "",
             f"同一份剧本（{len(rows)} 行，有字要念的 {len(spoken)} 行，其中不走讲述模式、vec 起作用的 {len(acted)} 行），"
             f"留 why 和去 why 各让 `claude -p`（{MODEL}）标 {n} 遍。剧本里都只存 vec、tempo、gain。", "",
             "## 两遍之间差多少", "",
             "vec 差：8 维逐维差的绝对值加起来，按行平均；主情绪不同：最大的那一维不一样（全 0 算一种）；"
             "tempo、gain 差：绝对值按行平均。同一种做法标两遍的差就是 `claude -p` 本身的出入。", "",
             "| 哪两遍 | vec 差 | 主情绪不同 | tempo 差 | gain 差 (dB) |", "|---|---|---|---|---|",
             *[dist(*p) for p in same], *[dist(*p) for p in pairs if p not in same],
             *[dist("原版", t) for t in tags], "", "## 每遍的整体倾向", "",
             "| 哪一遍 | 情绪浓度（vec 总和，平均） | vec 全 0 | tempo 平均 | 放慢的（<0.95） | gain 平均 | 压轻的（<0） "
             "| 费用 | 耗时 | 输出 token（其中思考） |", "|---|---|---|---|---|---|---|---|---|---|"]
    meta = {m["run"]: m for m in read_jsonl(cost)} if cost.exists() else {}
    for t in [*tags, "原版"]:
        x, m = runs[t], meta.get(t)
        u = m["usage"] if m else {}
        lines.append(f"| {t} | {np.mean([sum(x[r['idx']]['vec']) for r in acted]):.3f} "
                     f"| {sum(sum(x[r['idx']]['vec']) == 0 for r in acted)}/{len(acted)} "
                     f"| {np.mean([x[r['idx']]['tempo'] for r in spoken]):.3f} "
                     f"| {sum(x[r['idx']]['tempo'] < 0.95 for r in spoken)} "
                     f"| {np.mean([x[r['idx']]['gain'] for r in spoken]):+.2f} | {sum(x[r['idx']]['gain'] < 0 for r in spoken)} "
                     + (f"| ${m['total_cost_usd']:.2f} | {m['duration_ms'] / 1000:.0f} 秒 "
                        f"| {u['output_tokens']}（{u.get('output_tokens_details', {}).get('thinking_tokens', '—')}） |"
                        if m else "| — | — | — |"))
    mean = lambda w, k: np.mean([runs[t][k]["vec"] for t in tags if t.startswith(w)], axis=0)  # noqa: E731
    far = sorted(acted, key=lambda r: -l1(mean("留", r["idx"]), mean("去", r["idx"])))[:12]
    lines += ["", "## 两种做法差得最多的句子", "",
              f"按 {n} 遍平均的 vec 算差，列前 {len(far)} 句；每格是 vec / tempo / gain，依据取留why-1。", "",
              "| # | 谁 | 情绪 | 表演提示 | 台词 | " + " | ".join(tags) + " | 依据 |", "|---|---|---|---|---|" + "---|" * (len(tags) + 1)]
    for r in far:
        cell = lambda m: f"{vec_text(m['vec'])} / ×{m['tempo']:g} / {m['gain']:+g}"  # noqa: E731
        lines.append(f"| {r['idx']} | {r['speaker']} | {r['mood']} | {r['instruct']} | {r['text'].replace('|', '｜')} | "
                     + " | ".join(cell(runs[t][r["idx"]]) for t in tags) + f" | {runs[tags[0]][r['idx']].get('why', '')} |")
    md = WHY_AB / f"{script}-对比.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(md)


if __name__ == "__main__":
    if sys.argv[1] == "标注":
        cmd_label(sys.argv[2])
    elif sys.argv[1] == "写回":
        cmd_write(sys.argv[2:])
    elif sys.argv[1] == "why":
        why_ab(sys.argv[2])
    elif sys.argv[1] == "时长":
        duration_ab(sys.argv[2])
    elif sys.argv[1] == "合集":
        sampler()
