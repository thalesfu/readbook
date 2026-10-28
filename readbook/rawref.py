"""原声参考（07 号票）：Qwen3 直接拿游戏里的日语原声做复刻参考，不再经过 VoxCPM2 生成的中文定妆音。

三种用法见 load()：原声加日文逐字稿带文本复刻（原声版，有日语口音，不可用）、只取拼好那段原声的说话人
向量（原声向量版）、取几十句原声的平均说话人向量（原声平均版）。

挑片段的规则对所有角色一样，不按人调：
- 候选：这个角色在 bakeoff.REF_ROUTES 三条路线里的有声台词，去掉小样和〈2002〉自己的句子；
- 锚点（平常）：只用 3–8 秒、日文里没有！？…ー～这类感叹、拖音、省略的平稳句。按音高、语速离这个角色
  中位数（全部候选算）的远近排序，音高权重 2，从最近的开始拼，凑够 10 秒、不超过 18 秒；
- 情绪参考：种类跟混合版（voices.yaml）一样。混合版里从同一情绪的原声做出来的（moods 型，春希的轻声），
  换成同一情绪的原声，标签和挑法沿用 moods.label、moods.pick；从锚点派生的情绪（依绪、武也）和春希的
  讲述没有原声可对应，落回锚点；
- 拼接：句间 0.35 秒静音，峰值归一到 0.89，末尾补 0.5 秒静音（防止生成开头复读参考的尾巴，
  见 research/qwen3-torch 调研笔记 §4.4）；逐字稿每句补句号；
- 目标音高：所挑片段的音高中位数；目标语速沿用混合版的换算：4.2 字/秒 × 这个角色的日语语速
  ÷ 几个角色日语语速的中位数，再限制在 3.3–5.0；
- 平均向量：同一类的全部候选（锚点是这个角色的全部候选，情绪是标了这种情绪的全部候选）里，固定种子
  随机取 MEAN_N 句。

books/wa2/voices/原声.yaml 只记语音文件名、日文逐字稿和目标，不存音频：原声不进 git（硬约束：不上传到
任何云服务）。合成时按文件名从语音包里现拼，缓存在 out/rawref/。参考的键名沿用 zh_audio、zh_text
（bakeoff 里到处在用），原声版里装的是日语原声和日文逐字稿。

    uv run python -m readbook.rawref    # 重建 原声.yaml；moods 型角色的情绪标签要调 claude -p，缓存在 out/rawref/
"""

import json
import re

import numpy as np
import soundfile as sf
import yaml

from . import moods, qc
from .bakeoff import ZH_RATE, sample_rows, voiced_lines
from .scriptize import load_cast
from .trial import BOOK, OUT, read_jsonl
from .voices import SR, VoicePak

VOICES = BOOK / "voices"
RAW_YAML = VOICES / "原声.yaml"
CACHE = OUT / "rawref"
CALM = re.compile(r"^[^！？!?…ー～〜♪―]+$")
SOURCE = {"春希": "moods", "雪菜": "single", "依绪": "anchor", "武也": "anchor"}  # 和 passage.VOICE_SOURCE 一致
TAIL = 0.5  # 参考末尾补的静音（秒）
MEAN_N = 60  # 原声平均版每个参考取多少句原声算平均向量


def held_out() -> set[str]:
    """小样和〈2002〉用到的语音文件，不拿来当参考。"""
    rows = sample_rows() + read_jsonl(BOOK / "script" / "2010-11-25 〈2002〉.jsonl")
    return {r["voice"] for r in rows if r.get("voice")}


def pool(vp: VoicePak, code: str, skip: set[str]) -> list[dict]:
    """这个角色的候选原声：1.5–12 秒、测得出音高和语速的句子。"""
    items, seen = [], set()
    for name, jp, cn in voiced_lines(code):
        if name in seen or name in skip:
            continue
        seen.add(name)
        a = vp.audio(name)
        dur = len(a) / SR
        if 1.5 <= dur <= 12 and (f := qc.f0_median(a)) and (r := qc.rate(a, jp)):
            items.append({"i": len(items), "file": name, "jp": jp, "cn": cn, "dur": round(dur, 2),
                          "f0": round(f, 1), "rate": round(r, 3), "calm": bool(CALM.match(jp))})
    return items


def pick_anchor(items: list[dict], span=(10.0, 18.0)) -> list[dict]:
    mf = float(np.median([x["f0"] for x in items]))
    mr = float(np.median([x["rate"] for x in items]))
    calm = [x for x in items if x["calm"] and 3.0 <= x["dur"] <= 8.0]
    calm.sort(key=lambda x: 2 * abs(np.log(x["f0"] / mf)) + abs(np.log(x["rate"] / mr)))
    out, total = [], 0.0
    for x in calm:
        if total + x["dur"] > span[1]:
            continue
        out.append(x)
        total += x["dur"] + 0.35
        if total >= span[0]:
            break
    return out


def labels(name: str, desc: str, items: list[dict], limit=300, chunk=100) -> dict[str, str]:
    """前 limit 句 2.5–9 秒的候选交给 LLM 按情绪分类（和 bakeoff.cmd_moods_label 同样的范围）；语音文件 → 情绪。"""
    path = CACHE / f"{name}-labels.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    sel = [x for x in items if 2.5 <= x["dur"] <= 9.0][:limit]
    tags = {}
    for k in range(0, len(sel), chunk):
        tags.update(moods.label(name, desc, [{"i": x["i"], "jp": x["jp"], "cn": x["cn"]} for x in sel[k:k + chunk]]))
    out = {x["file"]: tags.get(x["i"], "平常") for x in sel}
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def entry(clips: list[dict], rate: float, group: list[dict]) -> dict:
    """clips 拼成参考音；group 是同一类的全部候选，固定种子随机取 MEAN_N 句给原声平均版算平均向量。"""
    import random

    return {"files": [x["file"] for x in clips], "text": moods.join_text([x["jp"] for x in clips]),
            "f0": round(float(np.median([x["f0"] for x in clips])), 1), "rate": round(min(max(rate, 3.3), 5.0), 2),
            "平均向量": sorted(x["file"] for x in random.Random(0).sample(group, min(MEAN_N, len(group))))}


def build():
    cast = load_cast(BOOK / "cast.yaml")
    mixed = yaml.safe_load((VOICES / "voices.yaml").read_text(encoding="utf-8"))
    vp = VoicePak()
    skip = held_out()
    pools = {n: pool(vp, cast["characters"][n]["code"], skip) for n in SOURCE}
    jp_rate = {n: float(np.median([x["rate"] for x in p])) for n, p in pools.items()}
    jp_mid = float(np.median(list(jp_rate.values())))
    out = {}
    for n, items in pools.items():
        zh_rate = ZH_RATE * jp_rate[n] / jp_mid
        refs = {"平常": entry(pick_anchor(items), zh_rate, items)}
        if SOURCE[n] == "moods":
            tag = labels(n, cast["characters"][n]["desc"], items)
            for m in mixed[n]["定妆音"]:
                if m in ("平常", moods.NARRATION):
                    continue
                group = [x for x in items if tag.get(x["file"]) == m]
                if clips := moods.pick(group):
                    base = [x for x in items if tag.get(x["file"]) == "平常"]
                    rate = zh_rate * float(np.median([x["rate"] for x in clips])) / float(
                        np.median([x["rate"] for x in base]))
                    refs[m] = entry(clips, rate, group)
        out[n] = {"日语语速中位": round(jp_rate[n], 2), "候选句数": len(items), "参考": refs}
        print(n, {m: (len(r["files"]), r["f0"], r["rate"]) for m, r in refs.items()}, flush=True)
    head = "# 07 号票的原声参考，规则见 readbook/rawref.py。只记语音文件名和日文逐字稿，原声不进 git。\n"
    RAW_YAML.write_text(head + yaml.safe_dump(out, allow_unicode=True, sort_keys=False, width=200),
                        encoding="utf-8")


def load(variant: str = "原声版") -> dict:
    """原声.yaml → {角色: {情绪: 参考}}，形状和 passage.load_voices 一样；音频按需从语音包拼到 out/rawref/。

    - 原声版：带文本复刻（ICL），原声的参考码加日文逐字稿进上下文。会把日语的发音带进中文，不可用；
    - 原声向量版：只取拼好的那段原声的说话人向量（xvec），参考码和逐字稿不进上下文；
    - 原声平均版：说话人向量取"平均向量"里那些原声各自向量的平均（spk_files），不挑片段。"""
    data = yaml.safe_load(RAW_YAML.read_text(encoding="utf-8"))
    vp = None
    out = {}
    for n, c in data.items():
        out[n] = {}
        for m, r in c["参考"].items():
            wav = CACHE / f"{n}-{m}.wav"
            if not wav.exists():
                vp = vp or VoicePak()
                a = moods.concat([vp.audio(f) for f in r["files"]], SR)
                CACHE.mkdir(parents=True, exist_ok=True)
                sf.write(wav, np.concatenate([a, np.zeros(int(TAIL * SR), dtype=np.float32)]), SR)
            out[n][m] = {"mood": m, "zh_audio": str(wav), "zh_text": r["text"], "f0": r["f0"], "rate": r["rate"],
                         "xvec": variant != "原声版"}
            if variant == "原声平均版":
                out[n][m]["spk_files"] = r["平均向量"]
    return out


if __name__ == "__main__":
    build()
