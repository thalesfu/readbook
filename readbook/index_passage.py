"""IndexTTS 2.0 / 2.5 做一段（10 号票）：每个角色固定一个音色锚点，逐句按剧本的情绪给情感向量。

    uv run python -m readbook.index_passage 2.5 小样
    uv run python -m readbook.index_passage 2 "2010-11-25 〈2002〉"
    uv run python -m readbook.index_passage 2.5-plain 小样      # 对照：同样的句子不给情绪
    uv run python -m readbook.index_passage 2.5-ref 小样        # 本人有该情绪的定妆音就拿它当情感参考
    uv run python -m readbook.index_passage 2.5-原声平均锚点 小样  # 16 号票：用更像原声的那个锚点
    uv run python -m readbook.index_passage 2.5-原声平均锚点全员 小样  # 16 号票：四人都用原声平均锚点
    uv run python -m readbook.index_passage 2.5-原声平均锚点新规 小样  # 17 号票：锚点按补全的挑选规则重挑
    uv run python -m readbook.index_passage stats 小样 2 2.5      # 候选级的"情绪 × 音色"表
    uv run python -m readbook.index_passage pair 小样 2.5 2.5-原声平均锚点   # 两版同一句同一种子的候选逐个配对
    uv run python -m readbook.index_passage repick 2.5-原声平均锚点新规 小样   # 19 号票：现成候选重新挑选、混音

- 音色：每个角色只用音色锚点（voices.yaml 的「平常」）。IndexTTS 只取参考的前 15 秒，超长的
  在 15 秒内最后一个停顿处截断（anchor_clip）。-原声平均锚点版只换这一项：rawanchor 用原声平均版重做的锚点
  比旧定妆音更像原声的角色（这次只有春希）换成新锚点，-原声平均锚点全员版四人都换，-原声平均锚点新规版
  四人都换成 17 号票的规则重挑的锚点（rawanchor.ANCHORS17）；情感参考、情感向量、
  质检目标都不变；挑选时"像锚点"也改成和所用的锚点比；
- 情绪：EMOTION 表，2.0 和 2.5 共用。「平常」不给向量；讲述模式（叙述者的旁白和心里话、情绪平常）
  拿叙述者自己的讲述定妆音当情感参考，音色仍取锚点；
- 每句按固定种子生成 TAKES 个候选，全部留着做统计；读入时切掉首尾静音（read_take）。挑选按混合版锚点版的做法顺序过一遍：
  和锚点的 CAM++ ≥ SIM_LINE 且音高、语速、句内变调过质检就停，再按音高语速挑、语速拉回目标的
  0.85–1.2 倍（bakeoff.finish）。实际用到的候选数记在 qc.json，按它们的耗时折算生产速度；
- 合成在 IndexTTS 自己的环境里跑（indextts_batch.py），这边负责出任务、挑选、混音；
- 产物在 out/index/<版本>/<名字>/：raw/ 候选、wav/ 成品、qc.json、timing.jsonl、MP3、对照剧本。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from . import qc
from .bakeoff import SIM_LINE, finish, ok
from .config import INDEX_DIR
from .moods import NARRATION
from .passage import load_rows, load_voices, mix
from .scriptize import load_cast
from .synth import load_yaml, tts_text
from .trial import BOOK, OUT
from .voices import SR, to48k

WORK = OUT / "index"
TAKES = 4
MAX_REF = 15.0  # IndexTTS 只取参考音频的前 15 秒
BATCH = Path(__file__).with_name("indextts_batch.py")

# 情绪 → 8 维情感向量 [喜, 怒, 哀, 惧, 厌恶, 忧郁, 惊讶, 平静]，是送进 normalize_emo_vec（偏置 +
# 总和压到 0.8 以下）之前的值。取自 02 号票调研笔记 §5，都是推测，靠这张票验证。
# 「害羞」「轻声」在 8 维里没有对应，只能近似；配角的「激动」「低落」按兴奋、难过处理。
EMOTION = {
    "平常": None,
    "开心": [0.6, 0, 0, 0, 0, 0, 0, 0.2],
    "兴奋": [0.75, 0, 0, 0, 0, 0, 0.25, 0],
    "生气": [0, 0.75, 0, 0, 0.1, 0, 0, 0],
    "难过": [0, 0, 0.45, 0, 0, 0.3, 0, 0],
    "哭腔": [0, 0, 0.8, 0.1, 0, 0.1, 0, 0],
    "害羞": [0.3, 0, 0, 0.2, 0, 0, 0, 0.2],
    "冷淡": [0, 0, 0, 0, 0.2, 0.1, 0, 0.5],
    "惊讶": [0.1, 0, 0, 0.1, 0, 0, 0.7, 0],
    "轻声": [0, 0, 0, 0.1, 0, 0.25, 0, 0.4],
}
EMOTION["激动"], EMOTION["低落"] = EMOTION["兴奋"], EMOTION["难过"]
EMO_ALPHA = 1.0  # 全局强度旋钮；强弱已经写进向量
RAWAVG = "原声平均锚点"  # 16 号票的版本后缀
RAWAVG_ALL = "原声平均锚点全员"
RAWAVG_RULE = "原声平均锚点新规"  # 17 号票


def speaker_anchors(variant: str, voices: dict) -> dict[str, Path]:
    """角色 → 整段音色锚点：默认是 voices.yaml 的「平常」；-原声平均锚点版里，rawanchor 重做的锚点
    比旧定妆音更像原声的角色换成新的（rawanchor.adopted），其余不变；-原声平均锚点全员版四人都换；
    -原声平均锚点新规版四人都换成 17 号票重挑的。"""
    old = {n: Path(v["平常"]["zh_audio"]) for n, v in voices.items()}
    if variant in (RAWAVG, RAWAVG_ALL, RAWAVG_RULE):
        from .rawanchor import ANCHORS, ANCHORS17, adopted

        d = ANCHORS17 if variant == RAWAVG_RULE else ANCHORS
        new = adopted() if variant == RAWAVG else {n: d / f"{n}.wav" for n in old}
        return {**old, **{n: p for n, p in new.items() if n in old}}
    return old


def clipped_anchors(variant: str, voices: dict) -> dict[str, Path]:
    """IndexTTS 实际用的说话人参考（前 15 秒）。"""
    sub = f"anchors-{variant}" if variant in (RAWAVG, RAWAVG_ALL, RAWAVG_RULE) else "anchors"
    return {n: anchor_clip(p, WORK / sub / f"{n}.wav") for n, p in speaker_anchors(variant, voices).items()}


def anchor_clip(src: Path, dst: Path) -> Path:
    """超过 15 秒的锚点在 10–15 秒之间能量最低的地方截断（句间停顿），尾部淡出 20ms。"""
    a, sr = sf.read(src, dtype="float32")
    if len(a) / sr > MAX_REF:
        hop, win = int(0.01 * sr), int(0.15 * sr)
        e = np.convolve(a[: int(MAX_REF * sr)] ** 2, np.ones(win) / win, "same")[::hop]
        lo = int(10.0 / 0.01)
        cut = (lo + int(np.argmin(e[lo:]))) * hop
        a = a[:cut].copy()
        fade = int(0.02 * sr)
        a[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, a, sr)
    return dst


def plan_row(row: dict, cast: dict, voices: dict, anchors: dict, mood_ref: bool = False) -> dict:
    """一句的合成设置：音色锚点 + 情绪（向量，或讲述定妆音），以及质检用的目标音高、语速。
    mood_ref（-ref 版）：本人有这种情绪的定妆音时拿它当情感参考，没有才用向量。"""
    n = row["speaker"]
    mood = row.get("mood", "平常")
    narration = n == cast["narrator"] and row["type"] in ("旁白", "心理活动")
    if narration and mood == "平常" and NARRATION in voices[n]:
        ref = voices[n][NARRATION]
        emo = {"emo_audio": str(Path(ref["zh_audio"]).resolve()), "how": NARRATION}
    elif mood_ref and mood != "平常" and mood in voices[n]:
        ref = voices[n][mood]
        emo = {"emo_audio": str(Path(ref["zh_audio"]).resolve()), "how": f"{mood}·定妆音"}
    else:
        ref = voices[n]["平常"]
        vec = EMOTION.get(mood, None)
        emo = {"emo_vector": vec, "how": mood if vec else "平常"}
    return {"spk": str(anchors[n].resolve()), **emo,
            "f0": ref["f0"], "rate": min(max(ref["rate"], 3.3), 5.0)}


def build(version: str, name: str):
    model, _, variant = version.partition("-")
    plain, mood_ref = variant == "plain", variant == "ref"
    rows = load_rows(name, None, None, None)
    cast = load_cast(BOOK / "cast.yaml")
    voices = load_voices()
    lex = load_yaml(BOOK / "lexicon.yaml")
    anchors = clipped_anchors(variant, voices)
    out = WORK / version / name
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    jobs = []
    for r in rows:
        text = tts_text(r, lex)
        if not qc.SPOKEN.search(text) or r["speaker"] not in voices:
            continue
        p = plan_row(r, cast, voices, anchors, mood_ref)
        if plain:  # 对照组：只做会加情绪的句子，情绪全去掉
            if p["how"] == "平常":
                continue
            p = {**p, "emo_vector": None, "emo_audio": None}
        chars = len(qc.SPOKEN.findall(text))
        for k in range(TAKES):
            jobs.append({"out": str((raw / f"{r['idx']:04d}-{k}.wav").resolve()), "text": text,
                         "spk": p["spk"], "emo_vector": p.get("emo_vector"), "emo_audio": p.get("emo_audio"),
                         "emo_alpha": EMO_ALPHA, "seed": r["idx"] * 100 + k, "chars": chars,
                         "max_secs": 3.5 + 0.5 * chars})
    # 按参考排好：IndexTTS 只缓存一个说话人和一个情感参考，换参考要重新提特征
    jobs.sort(key=lambda j: (j["spk"], j["emo_audio"] or "", j["out"]))
    (raw / "jobs.json").write_text(json.dumps({"version": model, "jobs": jobs}, ensure_ascii=False, indent=0),
                                   encoding="utf-8")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1"}
    subprocess.run(["uv", "run", "--project", str(INDEX_DIR), "python", str(BATCH), str(raw / "jobs.json")],
                   cwd=INDEX_DIR, env=env, check=True)
    if plain:
        return
    pick(version, name, rows, cast, voices, anchors, lex, mood_ref, speaker_anchors(variant, voices))


def repick(version: str, name: str):
    """19 号票：不重新合成，拿这一版 raw/ 里现成的候选重新挑选、混音（转 48 kHz 改用 soxr 以后），
    产物写到 <版本>-重采样/<名字>/。"""
    variant = version.partition("-")[2]
    voices = load_voices()
    pick(version, name, load_rows(name, None, None, None), load_cast(BOOK / "cast.yaml"), voices,
         clipped_anchors(variant, voices), load_yaml(BOOK / "lexicon.yaml"), variant == "ref",
         speaker_anchors(variant, voices), out=WORK / f"{version}-重采样" / name)


def pick(version, name, rows, cast, voices, anchors, lex, mood_ref=False, full=None, out=None):
    """full：挑选时"像锚点"和谁比（整段锚点），默认是 voices.yaml 的「平常」。
    out：成品写到哪（默认和候选同一个目录），候选总是读这一版的 raw/。qc.json 的 take 是挑中的第几个候选。"""
    from .speaker import SpeakerSim

    full = full or speaker_anchors("", voices)
    out = out or WORK / version / name
    raw, wav = WORK / version / name / "raw", out / "wav"
    wav.mkdir(parents=True, exist_ok=True)
    timing = {Path(t["out"]).name: t for t in map(json.loads, (raw / "timing.jsonl").read_text(encoding="utf-8")
                                                  .splitlines()) if "out" in t}
    sp = SpeakerSim()
    log = {}
    for r in rows:
        f = wav / f"{r['idx']:04d}.wav"
        text = tts_text(r, lex)
        if not qc.SPOKEN.search(text) or r["speaker"] not in voices:
            sf.write(f, np.zeros(int(0.8 * SR), dtype=np.float32), SR)
            continue
        p = plan_row(r, cast, voices, anchors, mood_ref)
        e_anc = sp.embed(full[r["speaker"]])
        cands = []
        for k in range(TAKES):
            a = read_take(raw / f"{r['idx']:04d}-{k}.wav")
            cands.append((sp.sim(sp.embed_audio(a, SR), e_anc) if len(a) > SR // 2 else 1.0, a))  # 太短的测不准
        used = []  # 按生产时的顺序过一遍：过关就停
        for s, a in cands:
            used.append((s, a))
            if s >= SIM_LINE and ok(a, text, p, 1.45):
                break
        passed = [k for k, (s, _) in enumerate(used) if s >= SIM_LINE] or [max(range(len(used)),
                                                                                key=lambda k: used[k][0])]
        a, m = finish([used[k][1] for k in passed], text, p)
        m = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}
        m.update(take=passed[m.pop("pick")], how=p["how"], tries=len(used), sims=[round(s, 3) for s, _ in cands],
                 secs=round(sum(timing[f"{r['idx']:04d}-{k}.wav"]["secs"] for k in range(len(used))), 2),
                 dur=round(len(a) / SR, 2))
        sf.write(f, a, SR)
        log[str(r["idx"])] = m
        print(r["idx"], m, flush=True)
    (out / "qc.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    mix(f"{name}-IndexTTS-{out.parent.name}", rows, cast, wav, out)


def read_take(path: Path) -> np.ndarray:
    """读候选并切掉首尾静音（阈值同 qc.speech_seconds），两头各留 0.1 秒。
    IndexTTS 有时说完了还空转出一长段静音，一直到生成上限（试跑里 4 秒的话拖到 17 秒）。"""
    a, sr = sf.read(path, dtype="float32")
    a = to48k(a.mean(axis=1) if a.ndim > 1 else a, sr)
    idx = np.flatnonzero(np.abs(a) > 0.02 * (np.abs(a).max() + 1e-9))
    if not len(idx):
        return a
    pad = int(0.1 * SR)
    return a[max(idx[0] - pad, 0): idx[-1] + pad]


# ---------- 候选级统计 ----------

BOOK_CHARS = 1_680_000  # 全书字数，折算全书耗时用


def stats(name: str, versions: list[str]):
    """所有候选（不经挑选）按情绪分组的音色表现，以及和不给情绪的对照组逐句配对比较、速度。"""
    from .evaluate import SHORT, real_embeddings, voice_gate
    from .speaker import SpeakerSim

    rows = {r["idx"]: r for r in load_rows(name, None, None, None)}
    cast = load_cast(BOOK / "cast.yaml")
    voices = load_voices()
    anchors = {n: WORK / "anchors" / f"{n}.wav" for n in voices}
    sp = SpeakerSim()
    gates = {n: voice_gate(E) for n, E in real_embeddings(sp, cast, list(voices)).items()}
    e_anc = {v: {n: sp.embed(f) for n, f in speaker_anchors(v.partition("-")[2], voices).items()} for v in versions}

    idle = {}  # 版本 → 尾部空转（切掉的静音）超过 1 秒的候选数、候选总数

    def score(version):
        """{idx: [(像原声, 像锚点) …]}，短于 SHORT 秒的候选不算。"""
        out = {}
        for f in sorted((WORK / version / name / "raw").glob("*-*.wav")):
            i = int(f.stem.split("-")[0])
            a = read_take(f)
            n_idle, n_all = idle.get(version, (0, 0))
            idle[version] = (n_idle + (sf.info(f).duration - len(a) / SR > 1.0), n_all + 1)
            if qc.speech_seconds(a) < SHORT:
                continue
            e, n = sp.embed_audio(a, SR), rows[i]["speaker"]
            out.setdefault(i, []).append((float(e @ gates[n][0]), float(e @ e_anc[version][n])))
        return out

    def group(i, version):
        r = rows[i]
        g = r["speaker"] if r["type"] == "台词" else f"{r['speaker']}·讲述"
        return g, plan_row(r, cast, voices, anchors, version.endswith("-ref"))["how"]

    lines = [f"# {name}：IndexTTS 候选级统计", "",
             f"每句 {TAKES} 个候选全部计入（不经挑选），干声，只算去掉首尾静音后 ≥{SHORT} 秒的候选。"
             "过门槛 = 像原声 ≥ 本人音色门槛（原声中心尺度 P5）。", ""]
    scored = {v: score(v) for v in versions}
    for v in versions:
        if v.endswith("-plain"):
            continue
        lines += [f"## {v}：情绪 × 音色", "", "| 谁 | 情绪 | 候选数 | 过门槛 | 像原声 中位 | 像锚点 中位 | 像锚点 ≥0.65 |",
                  "|---|---|---|---|---|---|---|"]
        cells = {}
        for i, ss in scored[v].items():
            cells.setdefault(group(i, v), []).extend(ss)
        for (g, how), ss in sorted(cells.items(), key=lambda x: (x[0][0], -len(x[1]))):
            n = g.split("·")[0]
            gate = gates[n][1]["center_p5"]
            real, anc = [s for s, _ in ss], [s for _, s in ss]
            lines.append(f"| {g} | {how} | {len(ss)} | {sum(s >= gate for s in real)}/{len(ss)} "
                         f"| {np.median(real):.2f} | {np.median(anc):.2f} | {sum(s >= SIM_LINE for s in anc)}/{len(ss)} |")
        lines.append("")
        # -ref 版和同模型的对照组配对：加情绪的句子是同一批；换了说话人参考的版本不能和它配
        plain = scored.get(f"{v.split('-')[0]}-plain") if v.partition("-")[2] in ("", "ref") else None
        if plain:
            lines += [f"### {v}：加情绪 vs 不加（同一句配对，像原声 中位）", "",
                      "| 情绪 | 句数 | 不加 | 加 | 差 | 过门槛 不加 → 加 |", "|---|---|---|---|---|---|"]
            pairs = {}
            for i in plain:
                if i in scored[v]:
                    pairs.setdefault(group(i, v)[1], []).append(i)
            for how, ids in sorted(pairs.items(), key=lambda x: -len(x[1])):
                a = [np.median([s for s, _ in plain[i]]) for i in ids]
                b = [np.median([s for s, _ in scored[v][i]]) for i in ids]
                gate = lambda i: gates[rows[i]["speaker"]][1]["center_p5"]  # noqa: E731
                pa = sum(s >= gate(i) for i in ids for s, _ in plain[i])
                pb = sum(s >= gate(i) for i in ids for s, _ in scored[v][i])
                na, nb = sum(len(plain[i]) for i in ids), sum(len(scored[v][i]) for i in ids)
                lines.append(f"| {how} | {len(ids)} | {np.median(a):.2f} | {np.median(b):.2f} "
                             f"| {np.median(np.array(b) - np.array(a)):+.2f} | {pa}/{na} → {pb}/{nb} |")
            lines.append("")
        t = [json.loads(x) for x in (WORK / v / name / "raw" / "timing.jsonl").read_text(encoding="utf-8").splitlines()]
        gen = [x for x in t if "out" in x]
        secs, dur, chars = (sum(x[k] for x in gen) for k in ("secs", "dur", "chars"))
        qcj = json.loads((WORK / v / name / "qc.json").read_text(encoding="utf-8"))
        used_secs = sum(m["secs"] for m in qcj.values())
        used_chars = sum(len(qc.SPOKEN.findall(rows[int(i)]["text"])) for i in qcj)
        tries = [m["tries"] for m in qcj.values()]
        per_char = used_secs / used_chars
        lines += [f"### {v}：速度", "",
                  f"- 说完后空转出 >1 秒静音的候选 {idle[v][0]}/{idle[v][1]}（已切掉，但生成耗时算在内）；",
                  f"- 加载模型 {t[0]['load_secs']:.0f} 秒（不计入下面）；",
                  f"- 单个候选：{secs:.0f} 秒生成 {dur:.0f} 秒音频，实时率 {secs / dur:.2f}，每字 {secs / chars:.2f} 秒；",
                  f"- 按挑选规则（过关就停）实际要生成的：平均每句 {np.mean(tries):.2f} 个候选，"
                  f"每字 {per_char:.2f} 秒，全书 {BOOK_CHARS / 10000:.0f} 万字约 {per_char * BOOK_CHARS / 3600:.0f} 小时"
                  f"（{per_char * BOOK_CHARS / 86400:.1f} 天），不含 CAM++ 和质检。", ""]
    md = WORK / f"{name}-候选统计.md"
    md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(md)


def pair(name: str, a: str, b: str):
    """两版同一句、同一种子的候选逐个配对（不经挑选）。两版只差说话人参考时，差别就是参考带来的：
    像原声、过门槛、认人（离谁的原声中心最近）、和最像的另一个人差多少、音高、变速前的语速；
    再加挑选后的结果（用了几个候选、要拉快多少）和生成速度。"""
    from .evaluate import SHORT, real_embeddings, voice_gate
    from .speaker import SpeakerSim

    rows = {r["idx"]: r for r in load_rows(name, None, None, None)}
    cast = load_cast(BOOK / "cast.yaml")
    voices = load_voices()
    lex = load_yaml(BOOK / "lexicon.yaml")
    anchors = clipped_anchors("", voices)  # 只用来取情绪的做法（how），两版一样
    sp = SpeakerSim()
    gates = {n: voice_gate(E) for n, E in real_embeddings(sp, cast, list(voices)).items()}

    def take(f):
        r = rows[int(f.stem.split("-")[0])]
        x = read_take(f)
        out = {"sim": None, "near": None, "gap": None, "f0": qc.f0_median(x), "rate": qc.rate(x, tts_text(r, lex))}
        if qc.speech_seconds(x) >= SHORT:
            e, n = sp.embed_audio(x, SR), r["speaker"]
            sims = {m: float(e @ g[0]) for m, g in gates.items()}
            out.update(sim=sims[n], near=max(sims, key=sims.get) == n,
                       gap=sims[n] - max(v for m, v in sims.items() if m != n))
        return out

    data = {v: {f.stem: take(f) for f in sorted((WORK / v / name / "raw").glob("*-*.wav"))} for v in (a, b)}
    keys = sorted(set(data[a]) & set(data[b]))

    def label(k):
        r = rows[int(k.split("-")[0])]
        g = r["speaker"] if r["type"] == "台词" else f"{r['speaker']}·讲述"
        return g, plan_row(r, cast, voices, anchors)["how"]

    def med(v, ks, key, fmt):
        xs = [data[v][k][key] for k in ks if data[v][k][key] is not None]
        return format(np.median(xs), fmt) if xs else "—"

    def line(title, ks):
        both = [k for k in ks if data[a][k]["sim"] is not None and data[b][k]["sim"] is not None]
        gate = lambda k: gates[rows[int(k.split("-")[0])]["speaker"]][1]["center_p5"]  # noqa: E731
        diff = np.median([data[b][k]["sim"] - data[a][k]["sim"] for k in both]) if both else None
        return (f"| {title} | {len(ks)} | {med(a, both, 'sim', '.2f')} → {med(b, both, 'sim', '.2f')} "
                f"| {f'{diff:+.2f}' if diff is not None else '—'} "
                f"| {sum(data[a][k]['sim'] >= gate(k) for k in both)}/{len(both)} → "
                f"{sum(data[b][k]['sim'] >= gate(k) for k in both)}/{len(both)} "
                f"| {sum(data[a][k]['near'] for k in both)}/{len(both)} → {sum(data[b][k]['near'] for k in both)}/{len(both)} "
                f"| {med(a, both, 'gap', '+.2f')} → {med(b, both, 'gap', '+.2f')} "
                f"| {med(a, ks, 'f0', '.0f')} → {med(b, ks, 'f0', '.0f')} "
                f"| {med(a, ks, 'rate', '.2f')} → {med(b, ks, 'rate', '.2f')} |")

    head = ["| {} | 候选 | 像原声 中位 | 配对差 中位 | 过门槛 | 认人·原声中心 | 像自己 − 最像的别人 中位 "
            "| 音高 Hz 中位 | 语速 字/秒 中位（变速前） |", "|---|---|---|---|---|---|---|---|---|"]
    lines = [f"# {name}：{a} → {b}，候选逐个配对", "",
             f"同一句、同一种子的候选一一配对，每句 {TAKES} 个全部计入，不经挑选。像原声和过门槛只算去掉首尾静音后 "
             f"≥{SHORT} 秒的候选（两版都够长的才配）；音高、语速算全部候选（语速要 ≥4 个字）。语速是生成出来的原样，"
             "还没有拉回目标。", "", "## 按组", "", head[0].format("组"), head[1]]
    groups = {}
    for k in keys:
        g, how = label(k)
        groups.setdefault(g, []).append(k)
        groups.setdefault((g, how), []).append(k)
    lines += [line(g, ks) for g, ks in groups.items() if isinstance(g, str)]
    lines += [line("合计", keys), "", "## 按组和情绪", "", head[0].format("组 · 情绪"), head[1]]
    lines += [line(f"{g[0]} · {g[1]}", ks) for g, ks in sorted((x for x in groups.items() if isinstance(x[0], tuple)),
                                                                key=lambda x: (x[0][0], -len(x[1])))]

    lines += ["", "## 挑选后和速度", "",
              "| 版本 | 平均每句用几个候选 | 挑中的变速前语速 中位 | 要拉快的句子（变速前 <目标 0.85 倍） | 单个候选 实时率 | 每字（用到的候选） | 全书 |",
              "|---|---|---|---|---|---|---|"]
    for v in (a, b):
        qcj = json.loads((WORK / v / name / "qc.json").read_text(encoding="utf-8"))
        slow = sum(1 for i, m in qcj.items() if m.get("rate") and
                   m["rate"] < 0.85 * plan_row(rows[int(i)], cast, voices, anchors)["rate"])
        t = [json.loads(x) for x in (WORK / v / name / "raw" / "timing.jsonl").read_text(encoding="utf-8").splitlines()]
        gen = [x for x in t if "out" in x]
        used_chars = sum(len(qc.SPOKEN.findall(rows[int(i)]["text"])) for i in qcj)
        per_char = sum(m["secs"] for m in qcj.values()) / used_chars
        lines.append(f"| {v} | {np.mean([m['tries'] for m in qcj.values()]):.2f} "
                     f"| {np.median([m['rate'] for m in qcj.values() if m.get('rate')]):.2f} | {slow}/{len(qcj)} "
                     f"| {sum(x['secs'] for x in gen) / sum(x['dur'] for x in gen):.2f} | {per_char:.2f} 秒 "
                     f"| {per_char * BOOK_CHARS / 86400:.1f} 天 |")
    md = WORK / f"{name}-配对-{a}-{b}.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(md)


if __name__ == "__main__":
    if sys.argv[1] == "stats":
        stats(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == "pair":
        pair(*sys.argv[2:5])
    elif sys.argv[1] == "repick":
        repick(*sys.argv[2:4])
    else:
        build(sys.argv[1], sys.argv[2])
