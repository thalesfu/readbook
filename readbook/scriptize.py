"""剧本化：给每一句标上类别、说话人、声音效果和表演提示。

调用无界面模式的 Claude Code（`claude -p`），输出按 JSON schema 约束。
两种跑法：
- shortcut：说话人已知（游戏数据），LLM 只补类别、效果、提示，以及没有名字的段落归谁；
- blind：去掉全部名字，当普通小说让 LLM 自己判断说话人，用来测准确率。
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


def ask_claude(prompt: str, model: str | None = None) -> dict:
    cmd = ["claude", "-p", "--output-format", "json", "--no-session-persistence",
           "--json-schema", json.dumps(SCHEMA, ensure_ascii=False)]
    if model:
        cmd += ["--model", model]
    res = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", check=True)
    out = json.loads(res.stdout)
    data = out.get("structured_output") or json.loads(out["result"])
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


def load_cast(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
