"""原声平均锚点：用原声平均版重做每个角色的音色锚点，交给 IndexTTS-2.5 当说话人参考（16 号票）；
17 号票给挑选规则补上音色门槛和音高范围。

规则对所有角色一样，不按人调：
- 文本：和旧定妆音（voices.yaml 的「平常」）一样，套读音表；
- 合成：Qwen3 只用说话人向量，向量取本人 60 句原声的平均（07 号票的原声平均版，rawref.load），温度 0.6，
  固定种子，一批 TAKES 个候选，不变速；
- 挑选（17 号票）：候选要同时
  - 没读错：字错率不比已生成的候选里最低的高出 MAX_CER 以上（人名地名常被识别成同音字，比如峰城祭→封城祭，
    所以不用绝对值）；
  - 过质检：bakeoff.ok，时长、句内变调、音高语速离目标不太远，目标取原声.yaml 的「平常」；
  - 音高离目标不超过本人原声音高四分位距的一半（按对数；原声是下面那同一批 300 句）。锚点的音高会带进成品
    （16 号票：雪菜、依绪、武也的成品音高都跟着锚点走），而质检是音高、语速合起来打分，语速贴目标时音高高两成
    也能过（武也）。雪菜高 4% 的锚点还在她原声的正常范围里，这条拦不住也不该拦；
  - 过本人的音色门槛：像原声 ≥ 本人原声和原声中心相似度的 P5（evaluate.voice_gate，和逐句的门槛同一把尺子）；
  合格的里面取像原声最高的。都不合格就换种子再生成一批，有合格的就停，最多 MAX_BATCHES 批；还没有，就放宽质检
  和音高这两条，在没读错、过门槛的里面取像原声最高的，再没有就在全部候选里取（用户 2026-10-28 定的）。
  试过再加一条"音高抖动（qc.jitter）不超过本人原声的中位"，用户试听后去掉了：它和听到的电子颤音对不上，
  三个春希锚点上正好排反，按它换的锚点成品更抖（17 号票）。
  16 号票的规则（rule="16"）只有前两条、只生成一批，都不过就在全部候选里取；
- 量：像原声和 evaluate 是同一把尺子（CAM++ 和本人 300 句原声的中心比）。16 号票的 measure 另把新旧锚点都量整段和
  IndexTTS 实际用的前 15 秒（index_passage.anchor_clip），也给出全部候选的分布，免得只看挑出来的那个；
  另算每个锚点离谁的原声中心最近，看角色之间还分不分得开；
- 采用（16 号票）：每人用前 15 秒像原声更高的那个，新锚点不比旧定妆音更像就沿用旧的（用户 2026-10-28 定的）。
  记在 锚点.yaml 的「采用」里，index_passage 的 -原声平均锚点版按它取说话人参考（adopted）。
  后来试听全员换新锚点更好（-原声平均锚点全员版，16 号票的定论），「采用」只是中间那一版的记录。

产物：候选在 out/rawanchor/。挑中的锚点和对比表：17 号票的规则在 books/wa2/trials/anchor-rule/，
16 号票的在 books/wa2/trials/rawavg-anchor/（合成的声音，可以进 git）。

    uv run python -m readbook.rawanchor       # 17 号票的规则
    uv run python -m readbook.rawanchor 16    # 16 号票的规则，再量新旧锚点
"""

import json

import numpy as np
import soundfile as sf
import yaml

from . import qc, rawref
from .trial import BOOK, OUT
from .voices import SR

DEST = BOOK / "trials" / "rawavg-anchor"  # 16 号票
ANCHORS = DEST / "锚点"
DEST17 = BOOK / "trials" / "anchor-rule"  # 17 号票
ANCHORS17 = DEST17 / "锚点"
WORK = OUT / "rawanchor"
TAKES = 8
MAX_BATCHES = 4
MAX_CER = 0.1
SEED = 16
HEAD = {"16": "# 16 号票的原声平均锚点，规则见 readbook/rawanchor.py。f0、rate 是质检目标（原声.yaml 的「平常」）；\n"
              "# 像原声是前 15 秒的，采用＝新锚点比旧定妆音更像原声才用新的。\n",
        "17": "# 17 号票的原声平均锚点，规则见 readbook/rawanchor.py。f0、rate 是质检目标（原声.yaml 的「平常」）；\n"
              "# 门槛＝本人音色门槛，音高范围＝目标 ± 本人原声音高四分位距的一半；\n"
              "# 不合格＝没过的条件。\n"}


def make(rule: str = "17"):
    import torch

    from .asrcheck import TorchWhisper, cer
    from .bakeoff import TorchQwen, ok
    from .evaluate import real_embeddings, real_pitch, voice_gate
    from .passage import load_voices
    from .scriptize import load_cast
    from .speaker import SpeakerSim
    from .synth import load_yaml, tts_text

    voices = load_voices()
    names = list(voices)
    refs = rawref.load("原声平均版")
    lex = load_yaml(BOOK / "lexicon.yaml")
    cast = load_cast(BOOK / "cast.yaml")
    sp = SpeakerSim()
    model, whisper = None, TorchWhisper()
    gates = {n: voice_gate(E) for n, E in real_embeddings(sp, cast, names).items()}
    spans = {n: float(np.log(np.divide(*np.percentile(f, [75, 25]))) / 2) for n, f in real_pitch(cast, names).items()}
    anchors = ANCHORS17 if rule == "17" else ANCHORS
    WORK.mkdir(parents=True, exist_ok=True)
    anchors.mkdir(parents=True, exist_ok=True)
    out = {}
    for n in names:
        text = tts_text({"text": voices[n]["平常"]["zh_text"], "type": "旁白"}, lex)
        ref = refs[n]["平常"]
        center, gate = gates[n][0], gates[n][1]["center_p5"]
        lo, hi = ref["f0"] * float(np.exp(-spans[n])), ref["f0"] * float(np.exp(spans[n]))
        cands = []
        for b in range(MAX_BATCHES if rule == "17" else 1):
            files = [WORK / f"{n}-{b * TAKES + k}.wav" for k in range(TAKES)]
            if all(f.exists() for f in files):  # 候选已经生成过：只重新打分
                takes = [sf.read(f, dtype="float32")[0] for f in files]
            else:
                model = model or TorchQwen()
                torch.manual_seed(SEED + b)
                takes = model.takes(text, ref, TAKES, 0.6)
                for f, a in zip(files, takes):
                    sf.write(f, a, SR)
            for k, a in enumerate(takes, b * TAKES):
                hyp = whisper.transcribe_audio(a, SR)
                f, r = qc.f0_median(a), qc.rate(a, text)
                cands.append({"k": k, "像原声": round(float(sp.embed_audio(a, SR) @ center), 3),
                              "字错率": round(cer(text, hyp), 3), "质检": bool(ok(a, text, ref, 1.45)),
                              "秒": round(len(a) / SR, 2), "音高": round(float(f), 1) if f else None,
                              "语速": round(float(r), 2) if r else None, "识别": hyp})
            floor = min(c["字错率"] for c in cands)
            for c in cands:
                c["不合格"] = [why for why, bad in (
                    ("读错", c["字错率"] > floor + MAX_CER), ("质检", not c["质检"]),
                    ("音高", rule == "17" and not (c["音高"] and lo <= c["音高"] <= hi)),
                    ("门槛", rule == "17" and c["像原声"] < gate)) if bad]
            good = [c for c in cands if not c["不合格"]]
            print(n, "第", b + 1, "批", "合格", len(good), flush=True)
            if good:
                break
        relaxed = [c for c in cands if set(c["不合格"]) <= {"质检", "音高"}] if not good and rule == "17" else []
        best = max(good or relaxed or cands, key=lambda c: c["像原声"])
        a, _ = sf.read(WORK / f"{n}-{best['k']}.wav", dtype="float32")
        sf.write(anchors / f"{n}.wav", a, SR)
        out[n] = {"text": text, "选中": best["k"], "合格候选": len(good), "f0": ref["f0"], "rate": ref["rate"],
                  "门槛": round(gate, 3), "音高范围": [round(lo, 1), round(hi, 1)]}
        if not good:
            out[n]["放宽"] = "质检和音高" if relaxed else "全部"
        out[n]["候选"] = cands
        print(n, "选中", best["k"], best["像原声"], "合格", len(good), "/", len(cands), flush=True)
    save(out, rule)


def save(info: dict, rule: str = "16"):
    dest = DEST17 if rule == "17" else DEST
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "锚点.yaml").write_text(HEAD[rule] + yaml.safe_dump(info, allow_unicode=True, sort_keys=False, width=200),
                                    encoding="utf-8")


def adopted() -> dict:
    """采用新锚点的角色 → 新锚点（整段）；不在里面的沿用旧定妆音。"""
    info = yaml.safe_load((DEST / "锚点.yaml").read_text(encoding="utf-8"))
    return {n: ANCHORS / f"{n}.wav" for n, x in info.items() if x.get("采用") == "新"}


def measure():
    """新旧锚点像原声的对比表，写到 DEST/锚点对比.md。"""
    from .evaluate import real_embeddings, voice_gate
    from .index_passage import anchor_clip
    from .passage import load_voices
    from .scriptize import load_cast
    from .speaker import SpeakerSim

    voices = load_voices()
    names = list(voices)
    info = yaml.safe_load((DEST / "锚点.yaml").read_text(encoding="utf-8"))
    sp = SpeakerSim()
    cast = load_cast(BOOK / "cast.yaml")
    gates = {n: voice_gate(E) for n, E in real_embeddings(sp, cast, names).items()}
    centers = {n: g[0] for n, g in gates.items()}
    old = {n: voices[n]["平常"]["zh_audio"] for n in names}
    new = {n: ANCHORS / f"{n}.wav" for n in names}

    def emb(path, tag, n):
        full = sp.embed(path)
        clip = sp.embed(anchor_clip(path, WORK / "clip" / f"{tag}-{n}.wav"))
        return full, clip

    def props(path, text):
        a, sr = sf.read(path, dtype="float32")
        a = a.mean(axis=1) if a.ndim > 1 else a
        f, r = qc.f0_median(a, sr), qc.rate(a, text, sr)
        return f"{len(a) / sr:.1f} 秒、{f:.0f} Hz、{r:.2f} 字/秒"

    E = {(tag, n): emb(p[n], tag, n) for tag, p in (("旧", old), ("新", new)) for n in names}
    for n in names:
        o, w = (float(E[(tag, n)][1] @ centers[n]) for tag in ("旧", "新"))
        info[n].update(旧定妆音像原声=round(o, 3), 新锚点像原声=round(w, 3), 采用="新" if w > o else "旧")
    save(info)
    lines = ["# 原声平均锚点和旧定妆音：像不像原声", "",
             "16 号票的第一步：用原声平均版（Qwen3 只用本人 60 句原声的平均说话人向量）按旧定妆音的文本重做音色锚点，"
             "先量它本身是不是比旧定妆音更像原声。规则见 `readbook/rawanchor.py`。", "",
             "- 像原声：CAM++ 和本人 300 句原声的中心比，和 `readbook.evaluate` 是同一把尺子；",
             "- 前 15 秒：IndexTTS 只取参考的前 15 秒，超长的在 10–15 秒间最安静处截断（`index_passage.anchor_clip`），"
             "这是它实际听到的；",
             f"- 候选：每人 {TAKES} 个，固定种子；挑选＝没读错（字错率不比这批最低的高出 {MAX_CER} 以上）、过质检"
             "（时长、句内变调、音高语速离原声目标不太远）的里面取像原声最高的。候选的中位和最高值说明挑选规则之外的余地。", "",
             "## 像原声", "",
             "| 角色 | 门槛 | 旧定妆音 整段 / 前 15 秒 | 新锚点候选 中位（最低–最高） | 合格候选 | 挑中的新锚点 整段 / 前 15 秒 | 差（前 15 秒） |",
             "|---|---|---|---|---|---|---|"]
    for n in names:
        c = centers[n]
        o, w = [float(e @ c) for e in E[("旧", n)]], [float(e @ c) for e in E[("新", n)]]
        sims = [x["像原声"] for x in info[n]["候选"]]
        lines.append(f"| {n} | {gates[n][1]['center_p5']:.2f} | {o[0]:.2f} / {o[1]:.2f} "
                     f"| {np.median(sims):.2f}（{min(sims):.2f}–{max(sims):.2f}） | {info[n]['合格候选']}/{len(sims)} "
                     f"| {w[0]:.2f} / {w[1]:.2f} | {w[1] - o[1]:+.2f} |")
    use = [n for n in names if info[n]["采用"] == "新"]
    lines += ["", f"**采用**：每人用前 15 秒像原声更高的那个。新锚点只有{'、'.join(use) or '（无）'}更像，"
              "其余沿用旧定妆音（`books/wa2/voices/<名字>/平常.wav`）。"]
    lines += ["", "## 时长、音高、语速（整段）", "",
              "| 角色 | 旧定妆音 | 新锚点 | 原声目标（原声.yaml） |", "|---|---|---|---|"]
    for n in names:
        lines.append(f"| {n} | {props(old[n], voices[n]['平常']['zh_text'])} | {props(new[n], info[n]['text'])} "
                     f"| {info[n]['f0']:.0f} Hz、{info[n]['rate']:.2f} 字/秒 |")
    lines += ["", "## 分不分得开（前 15 秒）", "",
              "每个锚点和四个人原声中心的相似度；粗体是最高的。差＝像自己 − 像最像的另一个人。", "",
              "| 锚点 | " + " | ".join(names) + " | 差 |", "|---" * (len(names) + 2) + "|"]
    for tag in ("旧", "新"):
        for n in names:
            s = {m: float(E[(tag, n)][1] @ centers[m]) for m in names}
            top = max(s, key=s.get)
            cells = [f"**{s[m]:.2f}**" if m == top else f"{s[m]:.2f}" for m in names]
            lines.append(f"| {tag}·{n} | " + " | ".join(cells) + f" | {s[n] - max(v for m, v in s.items() if m != n):+.2f} |")
    lines += ["", "锚点两两之间（前 15 秒）：", "", "| | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for tag in ("旧", "新"):
        for n in names:
            lines.append(f"| {tag}·{n} | " + " | ".join(
                "" if m == n else f"{float(E[(tag, n)][1] @ E[(tag, m)][1]):.2f}" for m in names) + " |")
    md = DEST / "锚点对比.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(md)
    print(json.dumps({n: [round(float(e @ centers[n]), 3) for e in E[("旧", n)] + E[("新", n)]] for n in names},
                     ensure_ascii=False))


def report17():
    """17 号票的规则挑了哪个、为什么：每人全部候选过了哪几条，和 16 号票挑中的比，写到 DEST17/锚点对比.md。"""
    new = yaml.safe_load((DEST17 / "锚点.yaml").read_text(encoding="utf-8"))
    old = yaml.safe_load((DEST / "锚点.yaml").read_text(encoding="utf-8"))
    lines = ["# 17 号票的锚点挑选规则：每个候选过了哪几条", "",
             f"候选的头一批（0–{TAKES - 1} 号）和 16 号票是同一批（原声平均版，固定种子），只换挑选规则。"
             "规则见 `readbook/rawanchor.py`：",
             "没读错、过质检、音高离目标不超过本人原声音高四分位距的一半、过本人音色门槛，合格的里面取像原声最高的；"
             f"都不合格就换种子再生成一批 {TAKES} 个，最多 {MAX_BATCHES} 批，还没有就放宽质检和音高。", "",
             "- 像原声：CAM++ 和本人 300 句原声的中心比，和 `readbook.evaluate` 同一把尺子；门槛是同一批原声的 P5；",
             "- 音高：`qc.f0_median`；原声音高的四分位距也取那 300 句；",
             "- 不合格一栏是没过的条件。", "",
             "## 挑中的", "",
             "| 角色 | 门槛 | 音高范围 Hz | 合格候选 | 16 号票挑中 | 17 号票挑中 |", "|---|---|---|---|---|---|"]

    def desc(x, k):  # 两版的候选是同一批，按编号取
        c = x["候选"][k]
        return f"{k} 号：像原声 {c['像原声']:.2f}、{c['音高']:.0f} Hz、{c['语速']:.2f} 字/秒"

    for n, x in new.items():
        why = {"质检和音高": "（没有合格的，放宽质检和音高）", "全部": "（没有合格的）"}.get(x.get("放宽"), "")
        lo, hi = x["音高范围"]
        lines.append(f"| {n} | {x['门槛']:.2f} | {lo:.0f}–{hi:.0f}（目标 {x['f0']:.0f}） "
                     f"| {x['合格候选']}/{len(x['候选'])} | {desc(x, old[n]['选中'])} | {desc(x, x['选中'])}{why} |")
    for n, x in new.items():
        lo, hi = x["音高范围"]
        lines += ["", f"## {n}", "", f"目标 {x['f0']:.0f} Hz、{x['rate']:.2f} 字/秒；音高范围 {lo:.0f}–{hi:.0f} Hz，"
                  f"门槛 {x['门槛']:.2f}。", "",
                  "| 候选 | 像原声 | 音高 Hz | 语速 | 字错率 | 不合格 | |", "|---|---|---|---|---|---|---|"]
        for c in x["候选"]:
            mark = "、".join(t for t, k in (("16 号票挑中", old[n]["选中"]), ("**17 号票挑中**", x["选中"])) if k == c["k"])
            lines.append(f"| {c['k']} | {c['像原声']:.2f} | {c['音高'] or 0:.0f} | {c['语速'] or 0:.2f} "
                         f"| {c['字错率']:.2f} | {'、'.join(c['不合格']) or '—'} | {mark} |")
    md = DEST17 / "锚点对比.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(md)


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["16"]:
        make("16")
        measure()
    else:
        make("17")
        report17()
