"""稳定性抽查：不经质检，每句一批生成几个候选，看有多少候选过音色门槛、字错率多少。

质检和重试会把跑偏的候选筛掉，成品的指标说的是"挑完以后"；这里看的是挑之前，同一份参考复刻得有多稳。
做法和 06 号票"不挑句子的对照"一样：小样或〈2002〉里四个人的每一句，按合成时同样的规则选参考，
一批生成 5 个候选；像原声、门槛和 evaluate 同一套，只算去掉首尾静音后 ≥1.5 秒的候选。

    uv run python -m readbook.stability 小样|2002 [--原声版|--原声向量版|--原声平均版] [每句候选数] [--只汇总]

结果在 out/stability/：<片段>-<版本>.jsonl（逐个候选）和 .md（汇总）。
"""

import json
import sys

import numpy as np

from . import qc
from .asrcheck import PUNCT, TorchWhisper, cer
from .bakeoff import load_qwen, qwen_takes
from .evaluate import PASSAGES, SHORT, real_embeddings, voice_gate
from .passage import VOICE_SOURCE, load_refs, pick_ref, variant_arg
from .scriptize import load_cast
from .speaker import SpeakerSim
from .synth import load_yaml, tts_text
from .trial import BOOK, OUT
from .voices import SR

TEMPERATURE = 0.6


def group(r: dict) -> str:
    return r["speaker"] if r["type"] == "台词" else f"{r['speaker']}·讲述"


def run(passage: str, variant: str, n: int) -> list[dict]:
    _, load_rows = PASSAGES[passage]
    cast = load_cast(BOOK / "cast.yaml")
    voices = load_refs(variant)
    lex = load_yaml(BOOK / "lexicon.yaml")
    names = list(VOICE_SOURCE)
    sp = SpeakerSim()
    gates = {k: voice_gate(E) for k, E in real_embeddings(sp, cast, names).items()}
    model, whisper = load_qwen(), TorchWhisper()
    recs = []
    for r in load_rows():
        text = tts_text(r, lex)
        if r["speaker"] not in names or not qc.SPOKEN.search(text):
            continue
        ref = pick_ref(r, cast, voices)
        center, stats = gates[r["speaker"]]
        for k, a in enumerate(qwen_takes(model, text, ref, n, TEMPERATURE)):
            rec = {"idx": r["idx"], "k": k, "speaker": r["speaker"], "type": r["type"], "group": group(r),
                   "ref": ref["mood"], "dur": round(qc.speech_seconds(a), 2),
                   "runaway": len(a) / SR > 2.5 + 0.5 * len(qc.SPOKEN.findall(text))}
            if rec["dur"] >= SHORT:
                s = float(sp.embed_audio(a, SR) @ center)
                rec |= {"sim_real": round(s, 3), "pass": s >= stats["center_p5"]}
            f, rt = qc.f0_median(a), qc.rate(a, text)
            rec |= {"f0": round(f, 1) if f else None, "rate": round(rt, 2) if rt else None}
            if len(PUNCT.sub("", text)) >= 2:
                hyp = whisper.transcribe_audio(a, SR)
                rec |= {"hyp": hyp, "cer": round(cer(text, hyp), 3)}
            recs.append(rec)
        line = [x for x in recs if x["idx"] == r["idx"]]
        print(r["idx"], r["speaker"], [x.get("sim_real") for x in line], [x.get("cer") for x in line], flush=True)
    return recs


def share(hits: list[bool]) -> str:
    return f"{sum(hits) / len(hits):.0%}" if hits else "—"


def summary(recs: list[dict]) -> list[str]:
    groups = [g for n in VOICE_SOURCE for g in (n, f"{n}·讲述") if any(r["group"] == g for r in recs)]
    out = [("| 组 | 句数 | 候选（≥1.5 秒） | 过门槛 | 像原声 中位（P10） | 有候选过门槛的句子 "
            "| 字错率 中位 | 字错率 >0.3 | 失控 |"),
           "|---|---|---|---|---|---|---|---|---|"]
    for g in groups + ["合计"]:
        sel = recs if g == "合计" else [r for r in recs if r["group"] == g]
        long_ = [r for r in sel if "sim_real" in r]
        sims = [r["sim_real"] for r in long_]
        lines = sorted({r["idx"] for r in long_})
        hit = sum(any(r["pass"] for r in long_ if r["idx"] == i) for i in lines)
        cers = [r["cer"] for r in sel if r.get("cer") is not None]
        sim = f"{np.median(sims):.2f}（{np.percentile(sims, 10):.2f}）" if sims else "—"
        cm = f"{np.median(cers):.3f}" if cers else "—"
        out.append(f"| {g} | {len({r['idx'] for r in sel})} | {len(long_)} | {share([r['pass'] for r in long_])} "
                   f"| {sim} | {hit}/{len(lines)} | {cm} | {share([c > 0.3 for c in cers])} "
                   f"| {sum(r['runaway'] for r in sel)} |")
    return out


def write(path, recs: list[dict], n: int):
    md = [f"# {path.name} 稳定性抽查", "", f"每句 {n} 个候选，温度 {TEMPERATURE}，不经质检。", ""] + summary(recs)
    path.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main(passage: str, *args: str):
    rest, variant = variant_arg([a for a in args if a != "--只汇总"])
    n = int(rest[0]) if rest else 5
    path = OUT / "stability" / f"{passage}-{variant}"
    if "--只汇总" in args:  # 已有逐候选记录，只重出汇总表
        recs = [json.loads(x) for x in path.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
        return write(path, recs, n)
    recs = run(passage, variant, n)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs),
                                          encoding="utf-8")
    write(path, recs, n)


if __name__ == "__main__":
    main(*sys.argv[1:])
