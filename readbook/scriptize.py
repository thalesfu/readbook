"""剧本化：给每一句标上类别、说话人、声音效果和表演提示。

调用无界面模式的 Claude Code（`claude -p`），输出按 JSON schema 约束。
两种跑法：
- shortcut：说话人已知（游戏数据），LLM 只补类别、效果、提示，以及没有名字的段落归谁；
- blind：去掉全部名字，当普通小说让 LLM 自己判断说话人，用来测准确率。

标完之后再单独问一遍停顿（pauses，18 号票）：这句前面是不是隐去了别人的话（比如单边电话只写这头），
要留几秒空白。单独问是为了不动已经标好的其他字段，也让它能看着标好的说话人判断。
"""

import json
import subprocess

import yaml

from .moods import MAIN

TYPES = ["旁白", "台词", "心理活动"]
EFFECTS = ["无", "电话", "回忆", "书信短信"]
MOODS = MAIN

SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer"},
                    "type": {"enum": TYPES},
                    "speaker": {"type": "string"},
                    "effect": {"enum": EFFECTS},
                    "instruct": {"type": "string"},
                    "mood": {"enum": MOODS},
                },
                "required": ["idx", "type", "speaker", "effect", "instruct", "mood"],
            },
        }
    },
    "required": ["lines"],
}

RULES = """\
你在为一部中文小说制作多角色有声书，任务是给每一行标注朗读方式。

## 类别（type）
- 旁白：叙述者的讲述——描写场景、动作、他人，交代事情经过。
- 台词：说出口的话。用眼神、表情传达的"对话"也算台词。
- 心理活动：某个角色没说出口的想法——此刻的感受、自问自答、想说却没说出口的话。

这部小说是第一人称，叙述者是「{narrator}」。没有引号的叙述段落要在「旁白」和「心理活动」之间二选一：
偏客观的讲述归旁白；直接吐露{narrator}此刻内心、带强烈主观情绪或自我对话的归心理活动。拿不准时归旁白。
旁白和{narrator}的心理活动，speaker 都填「{narrator}」。

## 声音效果（effect）
- 无：默认。
- 电话：这句话是经由电话传过来的声音。通话双方里，和叙述视角在同一处的那一方填「无」，电话另一头的填「电话」。
- 回忆：回想起来的、过去别人说过的话（不是当下发生的）。
- 书信短信：邮件、短信、信件、留言的内容。

## 表演提示（instruct）
一句不超过 15 个字的中文提示，说明这句该怎么念：情绪、语气、音量、节奏。例如「哽咽，断断续续」「故作轻松地调侃」「压低声音自语」。
旁白给平实的提示即可，比如「平静地讲述」。

## 情绪（mood）
从这些类别里选一个，决定合成时用这个角色哪一段情绪参考音：{moods}。
只看这一句说出来时的主要情绪；拿不准就选「平常」。旁白一律填「平常」。

## 角色
{cast}

## 输出
按 schema 返回 lines，idx 与输入一一对应，一行不漏。speaker 必须是上面角色表里的名字。
"""

SHORTCUT_NOTE = """\
下面每行已经给出了游戏数据里的说话人（name）。name 不为空且不是「？？？」的，speaker 照抄 name；
name 为空或「？？？」的，根据上下文判断是谁。"""

BLIND_NOTE = """\
下面每行都没有给出说话人，请根据上下文判断每一行是谁在说或在想。"""


def cast_text(cast: dict) -> str:
    rows = [f"- {name}：{c['desc']}" for name, c in cast["characters"].items()]
    return "\n".join(rows)


def build_prompt(lines: list[dict], cast: dict, blind: bool) -> str:
    head = RULES.format(narrator=cast["narrator"], cast=cast_text(cast), moods="、".join(MOODS))
    note = BLIND_NOTE if blind else SHORTCUT_NOTE
    body = []
    for x in lines:
        row = {"idx": x["idx"], "text": x["text"]}
        if not blind:
            row["name"] = x["speaker"]
        body.append(json.dumps(row, ensure_ascii=False))
    return f"{head}\n## 说明\n{note}\n\n## 输入\n" + "\n".join(body)


def ask_claude(prompt: str, model: str | None = None, schema: dict = SCHEMA, meta: dict | None = None) -> dict:
    """meta 给了就填进这次调用的用量（耗时、token、费用、模型），算成本用。"""
    cmd = ["claude", "-p", "--output-format", "json", "--no-session-persistence",
           "--json-schema", json.dumps(schema, ensure_ascii=False)]
    if model:
        cmd += ["--model", model]
    res = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", check=True)
    out = json.loads(res.stdout)
    data = out.get("structured_output") or json.loads(out["result"])
    if meta is not None:
        meta.update({k: out[k] for k in ("duration_ms", "total_cost_usd", "usage", "modelUsage") if k in out})
    return data


def scriptize(lines: list[dict], cast: dict, blind: bool = False, model: str | None = None) -> list[dict]:
    data = ask_claude(build_prompt(lines, cast, blind), model)
    got = {r["idx"]: r for r in data["lines"]}
    missing = [x["idx"] for x in lines if x["idx"] not in got]
    if missing:
        raise ValueError(f"LLM 漏了 {len(missing)} 行：{missing[:10]}")
    names = set(cast["characters"])
    by_code = {c["code"]: n for n, c in cast["characters"].items()
               if c.get("code") and c["code"] not in ("98", "99")}
    out = []
    for x in lines:
        r = got[x["idx"]]
        speaker = r["speaker"]
        if not blind:
            # 有专属语音编号的，以编号为准（顺带解决「？？？」）
            code = x.get("voice_code")
            if code in by_code:
                speaker = by_code[code]
            elif x["speaker"] and x["speaker"] != "？？？":
                speaker = x["speaker"]
        if speaker not in names:
            raise ValueError(f"第 {x['idx']} 行说话人「{speaker}」不在角色表里")
        row = {"idx": x["idx"], "type": r["type"], "speaker": speaker,
               "effect": r["effect"], "mood": r["mood"], "text": x["text"], "instruct": r["instruct"]}
        for k in ("voice", "jp"):
            if x.get(k):
                row[k] = x[k]
        out.append(row)
    return out


# ---------- 停顿：隐去的对白 ----------

MAX_PAUSE = 6.0

PAUSE_SCHEMA = {
    "type": "object",
    "properties": {
        "pauses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer"},
                    "pause": {"type": "number"},
                    "why": {"type": "string"},
                },
                "required": ["idx", "pause", "why"],
            },
        }
    },
    "required": ["pauses"],
}

PAUSE_RULES = """\
你在为一部中文小说制作多角色有声书。下面是已经标注好的剧本，每行给出类别（type）、说话人（speaker）、
声音效果（effect）和文字（text）。这部小说是第一人称，叙述者是「{narrator}」。

有的写法只写一方的话，把另一方的话隐去，读者靠上下文脑补对方说了什么。最常见的是单边电话：只写这头一句接一句，
那头的回答不写出来。配音时同一个人的几句话会紧挨着念，听起来像一口气说完。
你的任务：找出「这一行前面隐去了别人的话」的地方，标出在这一行前面要留几秒空白，让听众感觉到对方在说话。

## 怎么判断
- 只看这一行和前一行之间：前一行念完之后、这一行开口之前，有没有别人说了话却没写出来。
  线索：这一行在回应、反问、复述对方的话（「哎？」「这样啊」「是吗」「我？不…」），或者话题突然转了，
  而前一行里找不到这样说的来由。
- 不能只因为同一个人连着说就留白。同一个人自己接着说（犹豫、结巴、改口、补一句、一句话断成两行）、
  对白完整写出来的一来一回、旁白和心理活动的衔接，都不用留白。拿不准时不留。
- 只有省略号、没有字的行（如「………」）配音时是不到 1 秒的静音，表示这个人沉默，它本身也算一行。

## 留几秒（pause）
pause 是这一行前面总共要留的空白秒数。按被隐去的那段话大约多长来估，念的速度约每秒 4 个字：
- 对方只回了一两个字（「嗯」「对」）：约 1 秒；
- 对方回答了一两句：1.5–3 秒；
- 对方说了一段长的（解释、讲一件事）：3–4 秒；
- 中间还换了一通电话、过了一段时间、换了场景：在上面的基础上再多 1–2 秒，最多 {max_pause:g} 秒。
保留一位小数。

## 角色
{cast}

## 输出
按 schema 返回 pauses，只列出要留白的行，每行给一句不超过 30 字的理由（why）：对方大概说了什么、为什么留这么久。
不用留白的行不要列出。
"""


def build_pause_prompt(rows: list[dict], cast: dict) -> str:
    head = PAUSE_RULES.format(narrator=cast["narrator"], cast=cast_text(cast), max_pause=MAX_PAUSE)
    body = [json.dumps({k: r[k] for k in ("idx", "type", "speaker", "effect", "text")}, ensure_ascii=False)
            for r in rows]
    return f"{head}\n## 输入\n" + "\n".join(body)


def pauses(rows: list[dict], cast: dict, model: str | None = None) -> dict[int, dict]:
    """标好的剧本 → {idx: {"pause": 秒, "why": 理由}}，只含要留白的行；其余的 pause 都是 0。"""
    data = ask_claude(build_pause_prompt(rows, cast), model, PAUSE_SCHEMA)
    ids = {r["idx"] for r in rows}
    out = {}
    for p in data["pauses"]:
        if p["idx"] not in ids:
            raise ValueError(f"停顿标到了不存在的第 {p['idx']} 行")
        s = round(min(max(float(p["pause"]), 0.0), MAX_PAUSE), 1)
        if s > 0:
            out[p["idx"]] = {"pause": s, "why": p["why"]}
    return out


def with_pauses(rows: list[dict], marks: dict[int, dict]) -> list[dict]:
    """给每行补上 pause（没标的填 0），其他字段不动。"""
    return [{**r, "pause": marks[r["idx"]]["pause"] if r["idx"] in marks else 0} for r in rows]


def load_cast(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
