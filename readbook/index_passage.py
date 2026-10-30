"""IndexTTS 2.0 / 2.5 做一段（10 号票）：每个角色固定一个音色锚点，逐句按剧本的情绪给情感向量。

    uv run python -m readbook.index_passage 2.5-表演B混音 小样 "2010-11-25 〈2002〉"   # 定版做法，几段一起按轮生成
    uv run python -m readbook.index_passage 2.5-表演B混音 小样 --到 2.5-表演B混音-按轮   # 产物换个目录
    uv run python -m readbook.index_passage 对比 2.5-表演B混音 2.5-表演B混音-按轮 小样   # 换了候选的句子新旧对比
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
    uv run python -m readbook.index_passage 2.5-表演A 小样   # 11 号票：逐句表演设置（A/B/C 三种做法，见 perform）
    uv run python -m readbook.index_passage repick 2.5-表演B混音 小样 --到 2.5-表演B混音-重挑   # 现成候选按现在的规则重挑

- 音色：每个角色只用音色锚点（voices.yaml 的「平常」）。IndexTTS 只取参考的前 15 秒，超长的
  在 15 秒内最后一个停顿处截断（anchor_clip）。-原声平均锚点版只换这一项：rawanchor 用原声平均版重做的锚点
  比旧定妆音更像原声的角色（这次只有春希）换成新锚点，-原声平均锚点全员版四人都换，-原声平均锚点新规版
  四人都换成 17 号票的规则重挑的锚点（rawanchor.ANCHORS17）；情感参考、情感向量、
  质检目标都不变。-表演A/B/C 版的锚点同 -原声平均锚点新规；
- 情绪：EMOTION 表，2.0 和 2.5 共用。「平常」不给向量；讲述模式（叙述者的旁白和心里话、情绪平常）
  拿叙述者自己的讲述定妆音当情感参考，音色仍取锚点。定版（-表演B混音）的剧本行有表演设置（perform）的，
  情绪改用它的 vec，语速、音量用它的 tempo、gain（见 perform）；
- 按轮生成和挑选（合成规格第五、六节，生产 01 号票）：每句第 k 个候选的种子是 idx × 100 + k。几段剧本放在一起，
  第 k 轮只给还没过关的句子生成第 k 个候选，一轮调一次 IndexTTS（每次加载模型 20–30 秒），最多 TAKES 轮。
  每个候选读入时切掉首尾静音（read_take），有表演设置的先按语速变速，再判两样：
  - 音色门槛：去掉首尾静音后 ≥ evaluate.SHORT 秒的，和本人原声中心比，像原声 ≥ evaluate.voice_gate 的 center_p5；
    更短的声纹测不准，不查；
  - 质检：bakeoff.ok（时长、句内变调、音高语速偏差）。
  第一个两样都过的就停（more）。TAKES 轮都没过关的，在过了音色门槛的候选里挑偏差最小的（bakeoff.finish）；
  一个都没过音色门槛的再补最多 EXTRA 轮，有一个过了就停，还不过就取像原声最高的，记进复核清单。挑中的语速拉回
  目标的 0.85–1.2 倍。种子和顺序不变，所以挑出来的和"一次生成 TAKES 个再按顺序挑"一样；
- 合成在 IndexTTS 自己的环境里跑（indextts_batch.py），这边负责出任务、挑选、混音；
- 产物在 out/index/<版本>/<名字>/：raw/ 候选和 timing.jsonl、wav/ 成品、qc.json、MP3、对照剧本、复核清单；
  每轮的任务和加载耗时在 out/index/<版本>/轮次/，各轮的生成、加载、挑选耗时汇总在 轮次/耗时.jsonl。
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from . import perform, qc
from .bakeoff import SIM_LINE, finish, ok
from .config import INDEX_DIR
from .moods import NARRATION
from .passage import load_rows, load_voices, mix, stamp
from .scriptize import load_cast
from .synth import load_yaml, tts_text
from .trial import BOOK, OUT
from .voices import SR, to48k

WORK = OUT / "index"
TAKES = 4  # 按轮生成最多几轮（k = 0–3）
EXTRA = 4  # 一个候选都没过音色门槛的句子再补几轮（k = 4–7）
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
    -原声平均锚点新规版四人都换成 17 号票重挑的；-表演A/B/C 版同新规。"""
    variant = RAWAVG_RULE if variant in perform.VERSIONS else variant
    old = {n: Path(v["平常"]["zh_audio"]) for n, v in voices.items()}
    if variant in (RAWAVG, RAWAVG_ALL, RAWAVG_RULE):
        from .rawanchor import ANCHORS, ANCHORS17, adopted

        d = ANCHORS17 if variant == RAWAVG_RULE else ANCHORS
        new = adopted() if variant == RAWAVG else {n: d / f"{n}.wav" for n in old}
        return {**old, **{n: p for n, p in new.items() if n in old}}
    return old


def clipped_anchors(variant: str, voices: dict) -> dict[str, Path]:
    """IndexTTS 实际用的说话人参考（前 15 秒）。"""
    variant = RAWAVG_RULE if variant in perform.VERSIONS else variant
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


def plan(variant: str, name: str, cast: dict, voices: dict, anchors: dict, lex: dict) -> tuple[list[dict], dict]:
    """一段剧本：全部的行，以及要合成的句子 {idx: {row, text, chars, p, cands}}。p 是合成设置（plan_row，
    行里有表演设置 perform 的再套上 perform.apply），cands 按顺序放每个候选的判定（judge）。
    表演设置：定版（2.5-表演B混音）用剧本行里的，没有的照旧；A/B/C 试做版换成 11 号票那一份；
    其余版本去掉剧本里的，和以前的产物对得上。混音的音量也跟着行里的 perform 走。
    只有标点的、说话人没有音色的不合成，混音时插 0.8 秒静音。"""
    ver = perform.VERSIONS.get(variant)  # 11 号票：A/B/C，逐句表演设置
    rows = load_rows(name, None, None, None)
    if not ver:
        rows = [{k: v for k, v in r.items() if k != "perform"} for r in rows]
    elif variant not in perform.MIX:
        marks = perform.load(name)
        rows = [{**r, "perform": marks[r["idx"]]} for r in rows]
    todo = {}
    for r in rows:
        text = tts_text(r, lex)
        if not qc.SPOKEN.search(text) or r["speaker"] not in voices:
            continue
        p = plan_row(r, cast, voices, anchors, variant == "ref")
        if variant == "plain":  # 对照组：只做会加情绪的句子，情绪全去掉
            if p["how"] == "平常":
                continue
            p = {**p, "emo_vector": None, "emo_audio": None}
        if r.get("perform"):
            p = perform.apply(p, r["perform"], ver, variant in perform.MIX)
        todo[r["idx"]] = {"row": r, "text": text, "chars": len(qc.SPOKEN.findall(text)), "p": p, "cands": []}
    return rows, todo


def take_path(raw: Path, s: dict, k: int) -> Path:
    return (raw / f"{s['row']['idx']:04d}-{k}.wav").resolve()


def job(raw: Path, s: dict, k: int) -> dict:
    """一句的第 k 个候选交给 indextts_batch 的任务，种子 idx × 100 + k。"""
    p = s["p"]
    j = {"out": str(take_path(raw, s, k)), "text": s["text"], "spk": p["spk"], "emo_vector": p.get("emo_vector"),
         "emo_audio": p.get("emo_audio"), "emo_alpha": p.get("alpha", EMO_ALPHA), "seed": s["row"]["idx"] * 100 + k,
         "chars": s["chars"], "max_secs": 3.5 + 0.5 * s["chars"]}  # 生成上限管的是语义 token，duration_factor 不影响
    if p.get("emo_text"):
        j["emo_text"] = p["emo_text"]
    if p.get("df", 1.0) != 1.0:
        j["duration_factor"] = p["df"]
    return j


def synthesize(model: str, jobs: list[dict], path: Path) -> dict:
    """一轮：任务写进一份 jobs 文件，调一次 IndexTTS（已有的候选跳过）。返回这一轮新生成了几个、多少字，
    以及加载模型、生成（逐个候选合计）和整个子进程各花了几秒。"""
    # 按参考排好：IndexTTS 只缓存一个说话人和一个情感参考，换参考要重新提特征
    jobs = sorted(jobs, key=lambda j: (j["spk"], j["emo_audio"] or "", j["out"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": model, "jobs": jobs}, ensure_ascii=False, indent=0), encoding="utf-8")
    logs = {path.with_name("timing.jsonl")} | {Path(j["out"]).with_name("timing.jsonl") for j in jobs}
    seen = {f: len(f.read_text(encoding="utf-8").splitlines()) if f.exists() else 0 for f in logs}
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1"}
    t = time.perf_counter()
    subprocess.run(["uv", "run", "--project", str(INDEX_DIR), "python", str(BATCH), str(path)],
                   cwd=INDEX_DIR, env=env, check=True)
    wall = time.perf_counter() - t
    new = [json.loads(x) for f in logs if f.exists() for x in f.read_text(encoding="utf-8").splitlines()[seen[f]:]]
    made = [x for x in new if "out" in x]
    return {"made": len(made), "chars": sum(x["chars"] for x in made),
            "load": round(sum(x.get("load_secs", 0) for x in new), 1),
            "gen": round(sum(x["secs"] for x in made), 1), "wall": round(wall, 1)}


def take(raw: Path, s: dict, k: int) -> np.ndarray:
    """读第 k 个候选、切掉首尾静音；语速放混音的（表演设置的 tempo）先变速再挑。"""
    a = read_take(take_path(raw, s, k))
    return qc.stretch(a, s["p"]["tempo"]) if s["p"].get("tempo") else a


def judge(sp, a: np.ndarray, s: dict, gate: tuple[np.ndarray, dict]) -> dict:
    """一个候选：像原声（和本人原声中心的 CAM++ 相似度；去掉首尾静音后不到 SHORT 秒的测不准，不查，记 None）、
    过没过音色门槛（不查的算过）、过没过质检。gate 是 evaluate.voice_gate 的结果。"""
    from .evaluate import SHORT

    sim = None if qc.speech_seconds(a) < SHORT else float(sp.embed_audio(a, SR) @ gate[0])
    return {"sim": sim, "voice": sim is None or sim >= gate[1]["center_p5"],
            "ok": bool(ok(a, s["text"], s["p"], 1.45, slow=s["p"].get("slow", 1.0)))}


def more(cands: list[dict]) -> bool:
    """这句还要不要下一个候选：第一个音色门槛和质检都过的就停；TAKES 个都没过关、但有过音色门槛的，就在它们
    里面挑；一个都没过音色门槛的补轮次，补到有一个过了为止，最多 TAKES + EXTRA 个。"""
    if any(c["voice"] and c["ok"] for c in cands):
        return False
    if len(cands) < TAKES:
        return True
    return not any(c["voice"] for c in cands) and len(cands) < TAKES + EXTRA


def build(version: str, names: list[str], label: str | None = None):
    """几段剧本一起按轮生成，再逐段挑选、混音。产物在 out/index/<label>/<名字>/，label 默认是版本名。"""
    work = WORK / (label or version)
    run(version, names, work, work, generate=True)


def repick(version: str, names: list[str], label: str | None = None):
    """不重新合成，拿这一版 raw/ 里现成的候选按现在的规则、同样的顺序重新判、挑选、混音，缺的候选不补。
    产物写到 <label>/<名字>/，label 默认是 <版本>-重采样（19 号票：转 48 kHz 改用 soxr 以后）。"""
    run(version, names, WORK / version, WORK / (label or f"{version}-重采样"), generate=False)


def run(version: str, names: list[str], src: Path, dst: Path, generate: bool):
    """候选在 src/<名字>/raw/，产物写到 dst/<名字>/；耗时逐轮追加到 dst/轮次/耗时.jsonl。"""
    from .evaluate import real_embeddings, voice_gate
    from .speaker import SpeakerSim

    model, _, variant = version.partition("-")
    cast, voices, lex = load_cast(BOOK / "cast.yaml"), load_voices(), load_yaml(BOOK / "lexicon.yaml")
    anchors = clipped_anchors(variant, voices)
    plans = {n: plan(variant, n, cast, voices, anchors, lex) for n in names}
    raw = {n: src / n / "raw" for n in names}
    for d in raw.values():
        d.mkdir(parents=True, exist_ok=True)
    if variant == "plain":  # 对照组只做统计：每句 TAKES 个候选一次生成，不挑选
        synthesize(model, [job(raw[n], s, k) for n, (_, todo) in plans.items() for s in todo.values()
                           for k in range(TAKES)], src / "轮次" / "全部.json")
        return
    sp = SpeakerSim()
    gates = {n: voice_gate(E) for n, E in real_embeddings(sp, cast, list(voices)).items()}
    log = dst / "轮次" / "耗时.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)

    def record(rec: dict):
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(rec, flush=True)

    for k in range(TAKES + EXTRA):
        due = [(n, s) for n, (_, todo) in plans.items() for s in todo.values()
               if len(s["cands"]) == k and more(s["cands"]) and (generate or take_path(raw[n], s, k).exists())]
        if not due:
            break
        rec = {"round": k, "lines": len(due)}
        if generate:
            rec |= synthesize(model, [job(raw[n], s, k) for n, s in due], src / "轮次" / f"第{k}轮.json")
        t = time.perf_counter()
        for n, s in due:
            s["cands"].append(judge(sp, take(raw[n], s, k), s, gates[s["row"]["speaker"]]))
        record(rec | {"judge": round(time.perf_counter() - t, 1)})
    for n, (rows, todo) in plans.items():
        record({"name": n, **pick(version, n, rows, todo, cast, gates, raw[n], dst / n)})


def pick(version: str, name: str, rows: list[dict], todo: dict, cast: dict, gates: dict, raw: Path, out: Path) -> dict:
    """一段的挑选和混音，候选已经按 more 判好。在过了音色门槛的候选里挑偏差最小的（bakeoff.finish；过关就停的
    句子里通常就是最后那个过关的）；一个都没过的取像原声最高的。语速拉回目标，写 wav、qc.json、MP3、对照剧本、
    复核清单，返回挑选和混音各花了几秒。
    qc.json 的 take 是挑中第几个候选；sims 是每个候选的像原声（短句不查，记 null），voice、ok 是过没过音色门槛、
    质检，gate 是本人的音色门槛，fallback 表示补完轮次还没过音色门槛。有表演设置的句子另记 df、gain、target_rate。
    rows 是 plan 给的，带着这一版用的表演设置。"""
    ver = perform.VERSIONS.get(version.partition("-")[2])
    timing = {t["out"]: t for t in map(json.loads, (raw / "timing.jsonl").read_text(encoding="utf-8").splitlines())
              if "out" in t}
    emo_vecs = {}  # C 版的情感描述转成的向量：按轮生成的在 轮次/，以前的在 raw/
    for f in (raw / "emo_text.json", raw.parent.parent / "轮次" / "emo_text.json"):
        if f.exists():
            emo_vecs |= json.loads(f.read_text(encoding="utf-8"))
    wav = out / "wav"
    wav.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    log, notes = {}, {}
    for r in rows:
        f = wav / f"{r['idx']:04d}.wav"
        s = todo.get(r["idx"])
        if s is None:
            sf.write(f, np.zeros(int(0.8 * SR), dtype=np.float32), SR)
            continue
        cs, p = s["cands"], s["p"]
        passed = [k for k, c in enumerate(cs) if c["voice"]]
        fallback = not passed
        if fallback:  # 补完轮次还没过音色门槛：取像原声最高的，进复核清单
            passed = [max(range(len(cs)), key=lambda k: cs[k]["sim"])]
        a, m = finish([take(raw, s, k) for k in passed], s["text"], p)
        m = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}
        m.update(take=passed[m.pop("pick")], how=p["how"], tries=len(cs),
                 sims=[None if c["sim"] is None else round(c["sim"], 3) for c in cs],
                 voice=[c["voice"] for c in cs], ok=[c["ok"] for c in cs],
                 gate=round(gates[r["speaker"]][1]["center_p5"], 3), fallback=fallback,
                 secs=round(sum(timing[str(take_path(raw, s, k))]["secs"] for k in range(len(cs))), 2),
                 dur=round(len(a) / SR, 2))
        if r.get("perform"):
            notes[r["idx"]] = perform.note(r["perform"], p, ver, emo_vecs)
            m.update(df=p["df"], gain=r["perform"]["gain"], target_rate=round(p["rate"], 2))
        sf.write(f, a, SR)
        log[str(r["idx"])] = m
        print(r["idx"], m, flush=True)
    (out / "qc.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    secs = {"pick": round(time.perf_counter() - t, 1)}
    t = time.perf_counter()
    if notes:
        rows = [{**r, "perf_note": notes.get(r["idx"], "")} for r in rows]
    title = f"{name}-IndexTTS-{out.parent.name}"
    starts = mix(title, rows, cast, wav, out)
    write_review(title, rows, starts, log, out)
    return secs | {"mix": round(time.perf_counter() - t, 1)}


def write_review(title: str, rows: list[dict], starts: list[float], log: dict, out: Path):
    """复核清单的音色部分（合成规格第六节）：补完轮次还没过音色门槛、取了像原声最高的候选的句子。
    和对照剧本放在一起，时间对得上对照剧本。读音部分由生产 03 号票补。"""
    at = {r["idx"]: t for r, t in zip(rows, starts)}
    extra = [r for r in rows if log.get(str(r["idx"]), {}).get("tries", 0) > TAKES]
    bad = [r for r in extra if log[str(r["idx"])]["fallback"]]
    lines = [f"# {title} 复核清单", "", "要人去听的句子，时间同对照剧本。", "", "## 音色", "",
             f"{TAKES} 个候选都没过音色门槛的句子会补轮次，最多再生成 {EXTRA} 个：这一段补了 {len(extra)} 句，"
             f"补到过门槛的 {len(extra) - len(bad)} 句。下面是补完还没过的，用的是像原声最高的那个候选。", ""]
    if bad:
        lines += ["| 时间 | # | 谁 | 台词 | 像原声 | 音色门槛 |", "|---|---|---|---|---|---|"]
        for r in bad:
            m = log[str(r["idx"])]
            text = r["text"].replace("\n", " ").replace("|", "｜")
            lines.append(f"| {stamp(at[r['idx']])} | {r['idx']} | {r['speaker']} | {text} "
                         f"| {m['sims'][m['take']]:.3f} | {m['gate']:.3f} |")
    else:
        lines.append("没有。")
    (out / f"{title} 复核清单.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


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
    """所有候选（不经挑选）按情绪分组的音色表现，以及和不给情绪的对照组逐句配对比较、速度。
    只对每句 TAKES 个候选都生成了的版本有意义（按轮生成以前的试做）。"""
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


# ---------- 挑选规则新旧对比（生产 01 号票） ----------

COMPARE = BOOK / "trials" / "pick-rules"
HEED = {("2010-11-25 〈2002〉", 24): "用户之前听过、觉得 1.28 秒正好的是旧版"}  # 要特别听的句子


def why(m: dict) -> str:
    """qc.json 的一句：新规则下挑中的候选是怎么来的。"""
    k, n = m["take"], m["tries"]
    if m["fallback"]:
        return f"补到 {n} 个都没过音色门槛，取像原声最高的"
    if m["voice"][k] and m["ok"][k]:
        return f"第 {k} 号音色和质检都过" + ("，过关就停" if k == n - 1 else "，偏差最小")
    if n > TAKES:
        return f"前 {TAKES} 个都没过音色门槛，补到第 {n - 1} 号过了；质检没过，挑偏差最小的"
    return f"{n} 个都没过关，在过了音色门槛的里挑偏差最小的"


def compare(old: str, new: str, names: list[str]):
    """两版挑中的候选不一样的句子，每句连放两遍：旧 → 新，拼成一个 MP3 和一张时间表，放 trials/pick-rules/。
    混音同正式流程：后期链加表演设置的音量。"""
    from .evaluate import SHORT
    from .mix import db, process, to_mp3

    cast = load_cast(BOOK / "cast.yaml")
    gap_in, gap_out = 0.7, 2.0
    chunks, pos, table, same = [], 0.0, [], []
    for name in names:
        rows = {r["idx"]: r for r in load_rows(name, None, None, None)}
        q = {v: json.loads((WORK / v / name / "qc.json").read_text(encoding="utf-8")) for v in (old, new)}
        for i, m in q[new].items():
            r = rows[int(i)]
            if m["take"] == q[old][i]["take"]:
                if m["tries"] > TAKES:  # 补了轮次、挑中的没变：不放进 MP3，表后面交代一句
                    same.append(f"- {name.split(' ')[-1]} {i} 号{r['speaker']}「{r['text'].strip('“”')}」：{why(m)}"
                                f"（{max(s for s in m['sims'] if s is not None):.3f}，门槛 {m['gate']:.3f}），"
                                f"还是 {m['take']} 号候选，和旧版一样。")
                continue
            pan = cast["characters"].get(r["speaker"], {}).get("pan", 0.0)
            starts, durs = [], []
            for k, v in enumerate((old, new)):
                x = sf.read(WORK / v / name / "wav" / f"{int(i):04d}.wav", dtype="float32")[0]
                st = db(process(x, r, pan, True), m.get("gain", 0.0)).astype(np.float32)
                g = gap_in if k == 0 else gap_out
                starts.append(stamp(pos))
                durs.append(f"{qc.speech_seconds(x):.2f}")
                chunks += [st, np.zeros((int(g * SR), 2), dtype=np.float32)]
                pos += len(st) / SR + g
            sim = m["sims"][m["take"]]
            sim = "短句，不查" if sim is None else f"{sim:.3f}（门槛 {m['gate']:.3f}）"
            text = r["text"].replace("\n", " ").replace("|", "｜")
            table.append(f"| {starts[0]} | {starts[1]} | {name.split(' ')[-1]} | {i} | {r['speaker']} | {text} "
                         f"| {q[old][i]['take']} → {m['take']} | {durs[0]} → {durs[1]} 秒 | {sim} | {why(m)} "
                         f"| {HEED.get((name, int(i)), '')} |")
    out = np.concatenate(chunks)
    out *= min(1.0, 0.97 / float(np.abs(out).max()))
    COMPARE.mkdir(parents=True, exist_ok=True)
    mp3 = COMPARE / "新旧对比.mp3"
    to_mp3(out, mp3, {"album": "白色相簿2 剧本全集（试做）", "title": "挑选规则-新旧对比", "artist": "AI 有声书试做"})
    head = ["# 挑选规则新旧对比", "",
            f"新旧两版挑中的候选不一样的句子，每句连放两遍：旧版（`{old}`，像锚点 ≥{SIM_LINE}）→ 新版（`{new}`，"
            f"像原声 ≥ 本人音色门槛，去掉首尾静音后不到 {SHORT:g} 秒的不查）。同一句两遍隔 {gap_in:g} 秒，"
            f"换句隔 {gap_out:g} 秒。混音同正式流程。候选是第几号（种子 = 句号 × 100 + 候选号）；时长是挑中的候选"
            "语速拉回目标以后、加混音效果之前，有声部分（去掉首尾静音）的秒数。", "",
            "| 旧 | 新 | 片段 | # | 谁 | 台词 | 候选 | 时长 | 新版像原声 | 新规则怎么挑的 | 注意 |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    tail = ["", "补了轮次、挑中的没变的句子（没放进 MP3）：", "", *same] if same else []
    (COMPARE / "新旧对比.md").write_text("\n".join(head + table + tail) + "\n", encoding="utf-8", newline="\n")
    print(mp3, f"{len(out) / SR / 60:.1f} 分钟，{len(table)} 句")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    label = None
    if "--到" in args:  # 产物目录名
        i = args.index("--到")
        label, args = args[i + 1], args[:i] + args[i + 2:]
    if cmd == "stats":
        stats(args[0], args[1:])
    elif cmd == "pair":
        pair(*args[:3])
    elif cmd == "repick":
        repick(args[0], args[1:], label)
    elif cmd == "对比":
        compare(args[0], args[1], args[2:])
    else:
        build(cmd, args, label)
