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

生产 05 号票按全书给每条声线做锚点（make_lines），规则同 17 号票，对所有声线一样：
- 说话人向量、目标音高和语速、挑锚点片段：按声线取原声（rawref.line_ref）。目标语速的分母固定为 rawref.JP_MID；
- 文本：挑中片段的游戏中文译文，套读音表，再去掉读音表的拼音标注（plain，Qwen3 不认）；
- 门槛和音高范围：这条声线的原声（evaluate.real_clips，范围同 voicelines.gate_files，和声线库的音色门槛同一批）；
- 现有四人（春希、雪菜、依绪、武也）不重做：17 号票的锚点抄过来，逐句质检目标沿用 voices.yaml 的「平常」；
- 锚点放 books/wa2/voices/锚点/<声线名>.wav，登记进声线库（voicelines.ANCHOR_KEYS），挑选的明细在
  books/wa2/voices/锚点.yaml，汇总表在 锚点.md。候选在 out/anchors/。
新增的 7 个主要角色锚点先给用户试听（audition）：锚点加几句 IndexTTS-2.5 合成的平常台词，附原声对照。

    uv run python -m readbook.rawanchor       # 17 号票的规则
    uv run python -m readbook.rawanchor 16    # 16 号票的规则，再量新旧锚点
    uv run python -m readbook.rawanchor 声线 [声线名…]   # 生产 05 号票：全书声线的锚点（不给名字就是全部）
    uv run python -m readbook.rawanchor 试听              # 7 个新主要角色的试听，放 out/anchors/试听/（有原声，不进 git）
"""

import json
import re
import shutil
from functools import cache
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from . import qc, rawref, voicelines
from .trial import BOOK, OUT
from .voices import SR, VoicePak

DEST = BOOK / "trials" / "rawavg-anchor"  # 16 号票
ANCHORS = DEST / "锚点"
DEST17 = BOOK / "trials" / "anchor-rule"  # 17 号票
ANCHORS17 = DEST17 / "锚点"
PROD = voicelines.VOICES / "锚点"  # 生产 05 号票：全书声线的锚点
PROD_YAML = voicelines.VOICES / "锚点.yaml"
WORK05 = OUT / "anchors"
MAIN7 = ("和纱", "小春", "千晶", "麻理", "曜子", "孝宏", "朋")  # 新增的主要角色，锚点要先试听
PINYIN = re.compile(r"<([^<>|]+)\|[^<>]*>")  # 读音表的拼音标注 <字|拼音声调>（生产 03 号票加），Qwen3 不认
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


def choose(tag: str, text: str, ref: dict, center: np.ndarray, gate: float, lo: float, hi: float,
           qwen, whisper, sp, work: Path = WORK, rule: str = "17") -> tuple[np.ndarray, dict]:
    """一个锚点：按批生成候选、打分、挑选（规则见模块文档）。ref 是 Qwen3 的复刻参考，f0、rate 是质检目标；
    center、gate 是本人原声中心和音色门槛，lo–hi 是音高范围；qwen() 给出模型（用到才加载）。
    候选存在 work/<tag>-<k>.wav，已经生成过的只重新打分。返回挑中的音频，和 选中、合格候选、放宽（都不合格才有）、候选。"""
    import torch

    from .asrcheck import cer
    from .bakeoff import ok

    work.mkdir(parents=True, exist_ok=True)
    cands = []
    for b in range(MAX_BATCHES if rule == "17" else 1):
        files = [work / f"{tag}-{b * TAKES + k}.wav" for k in range(TAKES)]
        if all(f.exists() for f in files):  # 候选已经生成过：只重新打分
            takes = [sf.read(f, dtype="float32")[0] for f in files]
        else:
            model = qwen()
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
        print(tag, "第", b + 1, "批", "合格", len(good), flush=True)
        if good:
            break
    relaxed = [c for c in cands if set(c["不合格"]) <= {"质检", "音高"}] if not good and rule == "17" else []
    best = max(good or relaxed or cands, key=lambda c: c["像原声"])
    print(tag, "选中", best["k"], best["像原声"], "合格", len(good), "/", len(cands), flush=True)
    a, _ = sf.read(work / f"{tag}-{best['k']}.wav", dtype="float32")
    pick = {"选中": best["k"], "合格候选": len(good)}
    if not good:
        pick["放宽"] = "质检和音高" if relaxed else "全部"
    return a, pick | {"候选": cands}


def make(rule: str = "17"):
    from .asrcheck import TorchWhisper
    from .bakeoff import TorchQwen
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
    qwen, whisper = cache(TorchQwen), TorchWhisper()
    gates = {n: voice_gate(E) for n, E in real_embeddings(sp, cast, names).items()}
    spans = {n: float(np.log(np.divide(*np.percentile(f, [75, 25]))) / 2) for n, f in real_pitch(cast, names).items()}
    anchors = ANCHORS17 if rule == "17" else ANCHORS
    anchors.mkdir(parents=True, exist_ok=True)
    out = {}
    for n in names:
        text = tts_text({"text": voices[n]["平常"]["zh_text"], "type": "旁白"}, lex)
        ref = refs[n]["平常"]
        center, gate = gates[n][0], gates[n][1]["center_p5"]
        lo, hi = ref["f0"] * float(np.exp(-spans[n])), ref["f0"] * float(np.exp(spans[n]))
        a, pick = choose(n, text, ref, center, gate, lo, hi, qwen, whisper, sp, WORK, rule)
        sf.write(anchors / f"{n}.wav", a, SR)
        out[n] = {"text": text, "选中": pick["选中"], "合格候选": pick["合格候选"], "f0": ref["f0"], "rate": ref["rate"],
                  "门槛": round(gate, 3), "音高范围": [round(lo, 1), round(hi, 1)]}
        out[n] |= {k: pick[k] for k in ("放宽", "候选") if k in pick}
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


# ---------- 生产 05 号票：全书声线的锚点 ----------

def plain(text: str) -> str:
    """去掉读音表的拼音标注，只留字。"""
    return PINYIN.sub(r"\1", text)


def line_named(role: str, lines: dict) -> str:
    """主角色是 role 的那条声线。"""
    return next(n for n, v in lines.items() if next(iter(v["角色"])) == role)


def real(vp: VoicePak, emb, name: str, lines: dict) -> tuple[np.ndarray, float, np.ndarray]:
    """这条声线的原声中心、音色门槛（和声线库的同一批原声、同一个值），以及这批原声的音高（测不出的不计）。"""
    from .evaluate import real_clips, voice_gate

    E, f0 = [], []
    for f, a in real_clips(vp, lines[name]["编号"], files=voicelines.gate_files(name, lines)):
        E.append(emb(f, a))
        if x := qc.f0_median(a):
            f0.append(x)
    center, g = voice_gate(np.array(E))
    return center, g["center_p5"], np.array(f0)


def make_lines(names: list[str] | None = None):
    """给全书的声线（或 names 这几条）生成音色锚点，放 PROD，登记进声线库，明细写 PROD_YAML。
    现有四人不重做：17 号票的锚点抄过来。每做完一条就存一次，断了可以接着跑（候选已有的只重新打分）。"""
    from .asrcheck import TorchWhisper
    from .bakeoff import TorchQwen
    from .synth import load_yaml, tts_text

    lines = voicelines.load()
    names = names or list(lines)
    vp = VoicePak()
    emb = voicelines.Embeddings(vp)
    skip = rawref.held_out()
    lex = load_yaml(BOOK / "lexicon.yaml")
    old = yaml.safe_load((DEST17 / "锚点.yaml").read_text(encoding="utf-8"))
    mixed = yaml.safe_load((voicelines.VOICES / "voices.yaml").read_text(encoding="utf-8"))
    info = yaml.safe_load(PROD_YAML.read_text(encoding="utf-8")) if PROD_YAML.exists() else {}
    qwen, whisper = cache(TorchQwen), cache(TorchWhisper)
    PROD.mkdir(parents=True, exist_ok=True)
    for name in names:
        role = next(iter(lines[name]["角色"]))
        center, gate, f0 = real(vp, emb, name, lines)
        assert round(gate, 3) == lines[name]["音色门槛"], (name, gate)
        span = float(np.log(np.divide(*np.percentile(f0, [75, 25]))) / 2)
        if role in rawref.SOURCE:  # 现有四人：17 号票的锚点，逐句质检目标沿用 voices.yaml
            x, target = old[role], mixed[role]["定妆音"]["平常"]
            shutil.copyfile(ANCHORS17 / f"{role}.wav", PROD / f"{name}.wav")
            text, sim, f, r = x["text"], x["候选"][x["选中"]]["像原声"], target["f0"], target["rate"]
            info[name] = {"沿用": f"17 号票的锚点（{(ANCHORS17 / f'{role}.wav').relative_to(BOOK).as_posix()}，"
                                f"明细在同目录的 锚点.yaml）",
                          **{k: x[k] for k in ("text", "选中", "合格候选", "f0", "rate", "门槛", "音高范围")},
                          "挑中": {k: x["候选"][x["选中"]][k] for k in ("像原声", "音高", "语速", "秒")},
                          "候选数": len(x["候选"])}
        else:
            ref = rawref.line_ref(vp, name, skip)
            text = plain(tts_text({"text": ref["cn"], "type": "旁白"}, lex))
            lo, hi = ref["f0"] * float(np.exp(-span)), ref["f0"] * float(np.exp(span))
            q = {"zh_audio": name, "zh_text": text, "xvec": True, "spk_files": ref["平均向量"],
                 "f0": ref["f0"], "rate": ref["rate"]}
            a, pick = choose(name, text, q, center, gate, lo, hi, qwen, whisper(), emb.sp, WORK05)
            sf.write(PROD / f"{name}.wav", a, SR)
            sim, f, r = pick["候选"][pick["选中"]]["像原声"], ref["f0"], ref["rate"]
            info[name] = {"候选原声": ref["候选"], "候选原声句数": ref["候选句数"], "日语语速中位": ref["日语语速中位"],
                          "片段": ref["files"], "片段日文": ref["text"], "片段中文": ref["cn"],
                          "片段秒数": ref["片段秒数"], **({"片段放宽": ref["放宽"]} if "放宽" in ref else {}),
                          "平均向量句数": len(ref["平均向量"]), "text": text, "选中": pick["选中"],
                          "合格候选": pick["合格候选"], "f0": f, "rate": r, "门槛": round(gate, 3),
                          "音高范围": [round(lo, 1), round(hi, 1)],
                          **({"挑选放宽": pick["放宽"]} if "放宽" in pick else {}),
                          "候选": pick["候选"], "平均向量": ref["平均向量"]}
        lines[name] |= {"锚点": f"锚点/{name}.wav", "锚点文本": text, "锚点像原声": sim, "目标音高": f, "目标语速": r}
        voicelines.save(lines)
        save_lines(info, lines)
    emb.save()


def save_lines(info: dict, lines: dict):
    """明细按声线库的顺序写 PROD_YAML，汇总表写 锚点.md。"""
    info = {n: info[n] for n in lines if n in info}
    head = ("# 全书声线的音色锚点（生产 05 号票），规则见 readbook/rawanchor.py 和合成规格第四节。只记语音文件名，原声不进 git。\n"
            "# - 候选原声：取自三条路线，不到 60 句的取全书；片段：挑锚点片段的原声（文本、目标音高从它来），片段放宽＝凑不满\n"
            "#   10 秒放宽到了哪一档（rawref.RELAX）；平均向量：算说话人向量的原声；\n"
            "# - text：锚点念的字；f0、rate：目标音高、语速；门槛：音色门槛；音高范围：目标 ± 原声音高四分位距的一半；\n"
            "# - 候选：每批 8 个，不合格＝没过的条件；挑选放宽＝4 批都没有合格的，放宽到了哪一步；\n"
            "# - 沿用：现有四人的锚点是 17 号票的，没重做。\n")
    PROD_YAML.write_text(head + yaml.safe_dump(info, allow_unicode=True, sort_keys=False, width=200),
                         encoding="utf-8", newline="\n")
    md = ["# 全书声线的音色锚点", "",
          "生产 05 号票。规则见 `readbook/rawanchor.py` 和合成规格第四节，明细在 `锚点.yaml`，登记在 `声线.yaml`。", "",
          "- 原声：候选原声的范围和句数；片段：挑锚点片段的句数、秒数（放宽了的注明）；",
          "- 合格：合格候选 / 生成的候选；像原声、音高、语速是挑中的那个；目标是逐句质检的目标。", "",
          "| 声线 | 原声 | 片段 | 门槛 | 音高范围 Hz | 合格 | 选中 | 像原声 | 音高 Hz | 语速 | 目标 | 锚点文本 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, x in info.items():
        v = lines[n]
        if "沿用" in x:
            c, total, src, clip = x["挑中"], x["候选数"], "—", "沿用 17 号票"
        else:
            c, total, src = x["候选"][x["选中"]], len(x["候选"]), f"{x['候选原声']} {x['候选原声句数']}"
            clip = f"{len(x['片段'])} 句 {x['片段秒数']:.1f} 秒" + (f"（{x['片段放宽']}）" if "片段放宽" in x else "")
        lo, hi = x["音高范围"]
        got = f"{c['音高'] or 0:.0f} | {c['语速'] or 0:.2f}"
        why = f"（{x['挑选放宽']}）" if "挑选放宽" in x else ""
        md.append(f"| {n} | {src} | {clip} | {x['门槛']:.3f} | {lo:.0f}–{hi:.0f} | {x['合格候选']}/{total}{why} "
                  f"| {x['选中']} | {v['锚点像原声']:.2f} | {got} | {v['目标音高']:.0f} Hz、{v['目标语速']:.2f} | {v['锚点文本']} |")
    (voicelines.VOICES / "锚点.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")


def audition(roles: tuple[str, ...] = MAIN7, per: int = 4, takes: int = 4):
    """7 个新主要角色的锚点试听：每人先听锚点，再听 per 句 IndexTTS-2.5 合成的台词，每句前面放原声对照。
    - 台词：这条声线没挑进锚点的平稳句（3–8 秒、中文至少 8 个字），和挑锚点同样按音高、语速离中位数的远近排；
    - 合成照逐句合成的做法、不给情绪：锚点的前 15 秒当说话人参考，第 i 句第 k 个候选的种子是 (i + 1) × 100 + k，
      按顺序判，第一个过音色门槛和质检的就停；过了门槛的里面挑偏差最小的（bakeoff.finish），一个都没过取像原声最高的；
    - 产物在 out/anchors/试听/：每人一个 MP3（锚点，然后每句原声、合成），加一份 试听.md。有原声，不进 git。"""
    from .bakeoff import finish, ok
    from .index_passage import anchor_clip, read_take, synthesize
    from .mix import to_mp3
    from .synth import load_yaml, tts_text

    lines = voicelines.load()
    vp = VoicePak()
    emb = voicelines.Embeddings(vp)
    skip = rawref.held_out()
    lex = load_yaml(BOOK / "lexicon.yaml")
    dst = WORK05 / "试听"
    plans, jobs = {}, []
    for role in roles:
        name = line_named(role, lines)
        items, _ = rawref.line_pool(vp, name, skip)
        used = {x["file"] for x in rawref.pick_anchor(items)}
        mf, mr = (float(np.median([x[k] for x in items])) for k in ("f0", "rate"))
        says = [x for x in items if x["calm"] and 3.0 <= x["dur"] <= 8.0 and x["file"] not in used
                and len(qc.SPOKEN.findall(x["cn"])) >= 8]
        says = sorted(says, key=lambda x: 2 * abs(np.log(x["f0"] / mf)) + abs(np.log(x["rate"] / mr)))[:per]
        spk = anchor_clip(voicelines.anchor(name, lines), dst / "clip" / f"{name}.wav").resolve()
        center, gate, _ = real(vp, emb, name, lines)
        p = {"f0": lines[name]["目标音高"], "rate": lines[name]["目标语速"]}
        plans[role] = {"name": name, "center": center, "gate": gate, "p": p, "says": []}
        for i, x in enumerate(says):
            text = tts_text({"text": x["cn"], "type": "台词"}, lex)
            n = len(qc.SPOKEN.findall(text))
            outs = [dst / "raw" / f"{name}-{i}-{k}.wav" for k in range(takes)]
            jobs += [{"out": str(o.resolve()), "text": text, "spk": str(spk), "emo_vector": None, "emo_audio": None,
                      "emo_alpha": 1.0, "seed": (i + 1) * 100 + k, "chars": n, "max_secs": 3.5 + 0.5 * n}
                     for k, o in enumerate(outs)]
            plans[role]["says"].append({**x, "text": text, "outs": outs})
    (dst / "raw").mkdir(parents=True, exist_ok=True)
    if todo := [j for j in jobs if not Path(j["out"]).exists()]:
        synthesize("2.5", todo, dst / "jobs.json")
    md = ["# 新主要角色的锚点试听", "",
          "每人一个 MP3：先是锚点，然后每句台词先放原声、再放 IndexTTS-2.5 合成的（不给情绪）。原声只在本机，不进 git。", ""]
    gap = np.zeros(int(0.8 * SR), dtype=np.float32)

    def loud(a):
        return a / (np.abs(a).max() + 1e-9) * 0.89

    for no, (role, pl) in enumerate(plans.items(), 1):
        anchor, _ = sf.read(voicelines.anchor(pl["name"], lines), dtype="float32")
        parts = [loud(anchor), gap, gap]
        v = lines[pl["name"]]
        md += [f"## {no}. {role}（{pl['name']}）", "",
               f"锚点：像原声 {v['锚点像原声']:.2f}（门槛 {pl['gate']:.3f}），{len(anchor) / SR:.1f} 秒。{v['锚点文本']}", "",
               "| # | 原声 | 中文 | 候选 | 像原声 | 过门槛 | 过质检 |", "|---|---|---|---|---|---|---|"]
        for i, s in enumerate(pl["says"], 1):
            cands = []
            for o in s["outs"]:
                a = read_take(o)
                sim = float(emb.sp.embed_audio(a, SR) @ pl["center"]) if qc.speech_seconds(a) >= 1.5 else None
                cands.append({"a": a, "sim": sim, "voice": sim is None or sim >= pl["gate"],
                              "ok": bool(ok(a, s["text"], pl["p"], 1.45))})
                if cands[-1]["voice"] and cands[-1]["ok"]:
                    break
            passed = [c for c in cands if c["voice"]] or [max(cands, key=lambda c: c["sim"] or 0)]
            a, m = finish([c["a"] for c in passed], s["text"], pl["p"])
            c = passed[m["pick"]]
            parts += [loud(vp.audio(s["file"])), gap, loud(a), gap, gap]
            md.append(f"| {i} | {s['jp']} | {s['text']} | {len(cands)} | {c['sim'] or 0:.2f} "
                      f"| {'是' if c['voice'] else '否'} | {'是' if c['ok'] else '否'} |")
        md.append("")
        to_mp3(np.concatenate(parts), dst / f"{no}-{role}.mp3",
               {"album": "白色相簿2 锚点试听", "title": f"{no}-{role}", "artist": "AI 有声书试做"})
        print(dst / f"{no}-{role}.mp3", flush=True)
    (dst / "试听.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    emb.save()


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["16"]:
        make("16")
        measure()
    elif sys.argv[1:2] == ["声线"]:
        make_lines(sys.argv[2:] or None)
    elif sys.argv[1:] == ["试听"]:
        audition()
    else:
        make("17")
        report17()
