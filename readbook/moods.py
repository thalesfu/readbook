"""分情绪的定妆音。

每个角色每种情绪一段中文定妆音，合成时按（说话人, 情绪）选参考。流程：
1. label：把这个角色的候选原声（日文 + 中文）交给 LLM 按情绪分类；
2. build：从每类里挑音高居中的几句拼成日语参考，再用 VoxCPM2 加说话风格和情绪提示
   生成几段中文，挑音高最贴原声的一段作为定妆音。只生成实际用到的情绪。
"""

import json
import subprocess

import numpy as np
import soundfile as sf
import yaml

from . import qc

MAIN = ["平常", "开心", "兴奋", "生气", "难过", "哭腔", "害羞", "冷淡", "惊讶", "轻声"]
SUPPORT = ["平常", "激动", "低落"]
NARRATION = "讲述"
FALLBACK = {"哭腔": "难过", "难过": "平常", "惊讶": "兴奋", "兴奋": "开心", "开心": "平常",
            "害羞": "轻声", "轻声": "平常", "冷淡": "平常", "生气": "平常",
            "激动": "平常", "低落": "平常"}
TO_SUPPORT = {"开心": "激动", "兴奋": "激动", "生气": "激动", "惊讶": "激动",
              "难过": "低落", "哭腔": "低落"}
HINT = {
    "平常": "平常的语气",
    "开心": "开心，带着笑意",
    "兴奋": "兴奋，音量提高，语速加快",
    "生气": "生气，语气冲，咬字重",
    "难过": "难过，声音低沉无力",
    "哭腔": "带着哭腔，声音哽咽发颤",
    "害羞": "害羞，声音变小，有点结巴",
    "冷淡": "冷淡，平板，爱答不理",
    "惊讶": "惊讶，突然拔高",
    "轻声": "压低声音，轻声细语",
    "激动": "情绪激动，音量提高",
    "低落": "情绪低落，声音变弱",
    NARRATION: "平静低缓的讲述口吻，像在回忆往事，语速稍慢，声音贴近",
}


def allowed(tier: str) -> list[str]:
    return MAIN if tier == "主要" else SUPPORT if tier == "配角" else ["平常"]


def resolve(mood: str, tier: str, have: set[str]) -> str:
    """把 LLM 标的情绪落到这个角色实际有的定妆音上。"""
    if tier == "配角":
        mood = TO_SUPPORT.get(mood, mood if mood in SUPPORT else "平常")
    elif tier != "主要":
        mood = "平常"
    while mood not in have and mood in FALLBACK:
        mood = FALLBACK[mood]
    return mood if mood in have else "平常"


LABEL_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {"i": {"type": "integer"}, "mood": {"enum": MAIN}},
        "required": ["i", "mood"]}}},
    "required": ["items"],
}


def label(name: str, desc: str, lines: list[dict]) -> dict[int, str]:
    """lines: [{i, jp, cn}]；返回 i → 情绪。"""
    head = (f"下面是视觉小说角色「{name}」的台词（{desc}）。每行给出日文原文和中文译文。"
            f"按说这句话时的情绪，从这些类别里选一个：{'、'.join(MAIN)}。"
            "只看这句本身怎么说，拿不准就选平常。\n\n")
    body = "\n".join(json.dumps(x, ensure_ascii=False) for x in lines)
    cmd = ["claude", "-p", "--output-format", "json", "--no-session-persistence",
           "--json-schema", json.dumps(LABEL_SCHEMA, ensure_ascii=False)]
    res = subprocess.run(cmd, input=head + body, capture_output=True, text=True, check=True)
    out = json.loads(res.stdout)
    data = out.get("structured_output") or json.loads(out["result"])
    return {x["i"]: x["mood"] for x in data["items"]}


def pick(pool: list[dict], n_min=2, target=(6.0, 14.0)) -> list[dict] | None:
    """同一情绪里挑音高居中的几句，凑够 6–14 秒；不够 2 句返回 None。"""
    if len(pool) < n_min:
        return None
    med = float(np.median([p["f0"] for p in pool]))
    pool = sorted(pool, key=lambda p: abs(np.log(p["f0"] / med)))
    out, total = [], 0.0
    for p in pool:
        if total + p["dur"] > target[1]:
            continue
        out.append(p)
        total += p["dur"] + 0.35
        if total >= target[0]:
            break
    return out if len(out) >= n_min or total >= target[0] else None


def concat(clips: list[np.ndarray], sr: int, gap=0.35) -> np.ndarray:
    sil = np.zeros(int(gap * sr), dtype=np.float32)
    a = np.concatenate([x for c in clips for x in (c, sil)][:-1])
    return a / (np.abs(a).max() + 1e-9) * 0.89


def join_text(texts: list[str]) -> str:
    return "".join(t if t[-1] in "。！？…" else t + "。" for t in texts)


def best_take(model, text: str, jp_wav: str, instruct: str, f0: float, rate: float,
              tries: int = 4) -> tuple[np.ndarray, dict]:
    best = None
    for _ in range(tries):
        a = np.concatenate([np.array(x.audio, dtype=np.float32)
                            for x in model.generate(text=text, ref_audio=jp_wav, instruct=instruct)])
        score, m = qc.judge(a, text, f0, rate)
        score += max(0.0, qc.max_jump(a, f0) - 1.4) * 3
        if best is None or score < best[0]:
            best = (score, a, m)
    return best[1], {"score": round(best[0], 3), **best[2]}


def save_yaml(path, data):
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def write_wav(path, a, sr):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, a, sr)
