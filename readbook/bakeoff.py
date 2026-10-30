"""小样对比：同一段约 3 分钟的对话，用三种做法各合成一版。

共同的准备：
1. 代表性原声：不再挑"情绪平稳"的句子，而是挑音高、语速最接近这个角色中位数的几句，
   保留角色平时说话的样子；
2. 中文定妆音：日语原声 + 说话风格指令，用 VoxCPM2 生成几段中文，挑音高最贴原声、
   语速最贴目标的一段。之后三种做法都以它为参考，避开日语口音。

三种做法：
- vox：VoxCPM2 带文本复刻（定妆音同时当参考和续写提示）；
- qwen：Qwen3-TTS Base 带文本复刻；
- index：IndexTTS2（PyTorch/MPS，另一个环境），定妆音定音色，每句的表演提示定情绪。

三种做法都过同一道质检：音高或语速偏离角色目标太多就重做（最多 3 次取最好），
最后把语速拉回目标的 0.85–1.2 倍之间。

    uv run python -m readbook.bakeoff prepare
    uv run python -m readbook.bakeoff synth vox qwen
    uv run python -m readbook.bakeoff index      # 调另一个环境里的 IndexTTS2
    uv run python -m readbook.bakeoff mix
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from . import qc, wa2
from .config import INDEX_DIR, hf_model
from .scriptize import load_cast
from .synth import load_yaml, tts_text
from .trial import BOOK, OUT, read_jsonl
from .voices import SR, VoicePak, to48k

sys.path.insert(0, str(wa2.TOOLS))
from daily import ROUTES, walk  # noqa: E402
from lfpak import Pak  # noqa: E402

SEG = "02-2011-01-01"
PICK = list(range(0, 14)) + list(range(54, 85))
WORK = OUT / "bakeoff"
REF_ROUTES = ("01 第二部-共通", "02 第二部-雪菜线", "07 第三部-共通前半")
ZH_RATE = 4.2  # 普通话有声书的常见语速（字/秒），按角色原声的快慢再调
VOXCPM = "mlx-community/VoxCPM2-bf16"
QWEN_BASE = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
QWEN_TORCH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"  # Windows 上用的官方 PyTorch 版


def sample_rows() -> list[dict]:
    rows = [r for r in read_jsonl(BOOK / "script" / f"{SEG}.jsonl") if r["idx"] in PICK]
    for r in rows:
        r["effect"] = "无"  # 小样里去掉电话效果，好听清音色
    return rows


# ---------- 准备：代表性原声与中文定妆音 ----------

def voiced_lines(code: str):
    """(语音文件, 日文, 中文)，中日两份字节码按顺序对齐。"""
    jp, cn = Pak(wa2.GAME / "script.pak"), Pak(wa2.GAME / "ck-gal.pak")
    for folder, _, chain in ROUTES:
        if folder not in REF_ROUTES:
            continue
        for sid in chain:
            a = [e for e in walk(jp, sid, "cp932") if e[0] == "line"]
            b = [e for e in walk(cn, sid, "cn") if e[0] == "line"]
            for x, y in zip(a, b):
                if x[3] and x[3].endswith(f"_{code}.OGG"):
                    yield x[3], wa2.strip_quotes(wa2.clean(x[2]).replace("\n", "")), \
                        wa2.strip_quotes(wa2.clean(y[2]).replace("\n", ""))


def representative(vp: VoicePak, code: str, limit=200):
    """挑音高、语速最接近中位数的 3–4 句，拼成 10–18 秒。"""
    pool = []
    for name, jp, cn in voiced_lines(code):
        a = vp.audio(name)
        if not 3.0 <= len(a) / SR <= 8.0:
            continue
        f, r = qc.f0_median(a), qc.rate(a, jp)
        if f and r:
            pool.append((name, jp, cn, a, f, r))
        if len(pool) >= limit:
            break
    mf = float(np.median([p[4] for p in pool]))
    mr = float(np.median([p[5] for p in pool]))
    pool.sort(key=lambda p: abs(np.log(p[4] / mf)) * 2 + abs(np.log(p[5] / mr)))
    picked, total = [], 0.0
    for p in pool:
        d = len(p[3]) / SR
        if total + d > 18:
            continue
        picked.append(p)
        total += d + 0.35
        if total >= 10:
            break
    sil = np.zeros(int(0.35 * SR), dtype=np.float32)
    audio = np.concatenate([x for p in picked for x in (p[3], sil)][:-1])
    audio = audio / np.abs(audio).max() * 0.89
    end = lambda t: t if t[-1] in "。！？…" else t + "。"  # noqa: E731
    return {"audio": audio, "jp": "".join(end(p[1]) for p in picked),
            "cn": "".join(end(p[2]) for p in picked), "f0": mf, "jp_rate": mr,
            "files": [p[0] for p in picked]}


def cmd_prepare(names=("春希", "依绪", "武也", "雪菜"), tries=5):
    from mlx_audio.tts.utils import load_model

    cast = load_cast(BOOK / "cast.yaml")
    vp = VoicePak()
    refs = {}
    for n in names:
        c = cast["characters"][n]
        refs[n] = representative(vp, c["code"])
        print(n, f"原声 {len(refs[n]['audio']) / SR:.1f}s 音高 {refs[n]['f0']:.0f} 语速 {refs[n]['jp_rate']:.1f}", flush=True)
    jp_mean = float(np.median([r["jp_rate"] for r in refs.values()]))
    model = load_model(VOXCPM)
    (WORK / "refs").mkdir(parents=True, exist_ok=True)
    meta = {}
    for n, r in refs.items():
        jp_wav = WORK / "refs" / f"{n}-jp.wav"
        sf.write(jp_wav, r["audio"], SR)
        target_rate = ZH_RATE * r["jp_rate"] / jp_mean
        style = cast["characters"][n]["style"]
        best = None
        for i in range(tries):
            a = np.concatenate([np.array(x.audio, dtype=np.float32) for x in
                                model.generate(text=r["cn"], ref_audio=str(jp_wav), instruct=style)])
            score, m = qc.judge(a, r["cn"], r["f0"], target_rate)
            sf.write(WORK / "refs" / f"{n}-zh-{i}.wav", a, SR)
            print(f"  {n} 定妆候选 {i}：音高 {m['f0']} 语速 {m['rate']} 分 {score:.2f}", flush=True)
            if best is None or score < best[0]:
                best = (score, i, m)
        zh = WORK / "refs" / f"{n}-zh.wav"
        zh.write_bytes((WORK / "refs" / f"{n}-zh-{best[1]}.wav").read_bytes())
        meta[n] = {"zh_audio": str(zh), "zh_text": r["cn"], "jp_audio": str(jp_wav), "jp_text": r["jp"],
                   "f0": r["f0"], "rate": target_rate, "picked": best[1], "files": r["files"]}
    (WORK / "refs.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False), encoding="utf-8")


# ---------- 分情绪定妆音 ----------

MOOD_DIR = WORK / "moods"


def cmd_moods_label(names=("春希", "依绪", "武也", "雪菜"), limit=300, chunk=100):
    """给每个角色的候选原声标情绪，结果缓存在 moods/<名字>-labels.json。"""
    from concurrent.futures import ThreadPoolExecutor

    from . import moods

    cast = load_cast(BOOK / "cast.yaml")
    vp = VoicePak()
    MOOD_DIR.mkdir(parents=True, exist_ok=True)

    def one(n):
        path = MOOD_DIR / f"{n}-labels.json"
        if path.exists():
            return n
        c = cast["characters"][n]
        items = []
        for name, jp, cn in voiced_lines(c["code"]):
            a = vp.audio(name)
            dur = len(a) / SR
            if 2.5 <= dur <= 9.0 and (f := qc.f0_median(a)) and (r := qc.rate(a, jp)):
                items.append({"i": len(items), "file": name, "jp": jp, "cn": cn,
                              "f0": f, "rate": r, "dur": dur})
            if len(items) >= limit:
                break
        tags = {}
        for k in range(0, len(items), chunk):
            part = [{"i": x["i"], "jp": x["jp"], "cn": x["cn"]} for x in items[k:k + chunk]]
            tags.update(moods.label(n, c["desc"], part))
        for x in items:
            x["mood"] = tags.get(x["i"], "平常")
        path.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
        return n

    with ThreadPoolExecutor(len(names)) as ex:
        for n in ex.map(one, names):
            items = json.loads((MOOD_DIR / f"{n}-labels.json").read_text(encoding="utf-8"))
            from collections import Counter
            print(n, dict(Counter(x["mood"] for x in items)), flush=True)


def cmd_moods_build():
    """只为小样里实际用到的（说话人, 情绪）生成定妆音；春希另加一段讲述。"""
    from mlx_audio.tts.utils import load_model

    from . import moods

    cast = load_cast(BOOK / "cast.yaml")
    narrator = cast["narrator"]
    vp = VoicePak()
    rows = sample_rows()
    out = MOOD_DIR / "refs.yaml"
    refs = yaml.safe_load(out.read_text(encoding="utf-8")) if out.exists() else {}
    model = None
    for n in sorted({r["speaker"] for r in rows}):
        c = cast["characters"][n]
        items = json.loads((MOOD_DIR / f"{n}-labels.json").read_text(encoding="utf-8"))
        groups = {}
        for x in items:
            groups.setdefault(x["mood"], []).append(x)
        picked = {m: moods.pick(g) for m, g in groups.items()}
        have = {m for m, p in picked.items() if p}
        need = {moods.resolve(r["mood"], c["tier"], have) for r in rows
                if r["speaker"] == n and r["type"] == "台词"}
        if n == narrator:
            need |= {moods.NARRATION, "平常"}
        base = picked["平常"]
        base_f0 = float(np.median([p["f0"] for p in base]))
        base_rate = float(np.median([p["rate"] for p in groups["平常"]]))
        zh_base = refs.get(n, {}).get("_rate") or yaml.safe_load(
            (WORK / "refs.yaml").read_text(encoding="utf-8"))[n]["rate"]
        refs.setdefault(n, {})["_rate"] = zh_base
        refs[n]["_have"] = sorted(have)
        for m in sorted(need):
            if m in refs[n]:
                continue
            clips = base if m == moods.NARRATION else picked[m]
            jp_rate = float(np.median([p["rate"] for p in clips]))
            f0 = float(np.median([p["f0"] for p in clips]))
            rate = zh_base * jp_rate / base_rate * (0.9 if m == moods.NARRATION else 1.0)
            jp_wav = MOOD_DIR / f"{n}-{m}-jp.wav"
            moods.write_wav(jp_wav, moods.concat([vp.audio(p["file"]) for p in clips], SR), SR)
            text = moods.join_text([p["cn"] for p in clips])
            model = model or load_model(VOXCPM)
            a, info = moods.best_take(model, text, str(jp_wav),
                                      f"{c['style']}此刻：{moods.HINT[m]}。", f0, rate)
            zh = MOOD_DIR / f"{n}-{m}-zh.wav"
            moods.write_wav(zh, a, SR)
            refs[n][m] = {"zh_audio": str(zh), "zh_text": text, "f0": f0, "rate": rate,
                          "files": [p["file"] for p in clips],
                           **{k: float(v) for k, v in info.items() if v is not None}}
            print(n, m, f"音高 {f0:.0f} 语速 {rate:.1f}", info, flush=True)
            moods.save_yaml(out, refs)
    moods.save_yaml(out, refs)


def mood_ref(row: dict, refs: dict, cast: dict) -> dict:
    from . import moods

    n = row["speaker"]
    if n == cast["narrator"] and row["type"] in ("旁白", "心理活动"):
        ref = refs[n].get(moods.NARRATION) or refs[n]["平常"]  # 原声版没有讲述，用锚点
    else:
        m = moods.resolve(row.get("mood", "平常"), cast["characters"][n]["tier"], set(refs[n]["_have"]))
        ref = refs[n].get(m) or refs[n]["平常"]
    # 按日语原声换算的目标语速会失真（日文字数和中文对不上），限制在正常的中文语速范围
    return {**ref, "rate": min(max(ref["rate"], 3.3), 5.0)}


# ---------- 音色锚点与派生情绪（第三轮） ----------

ANCHOR_DIR = WORK / "anchor"
SIM_LINE = 0.65    # 逐句结果和锚点的 CAM++ 相似度下限
SIM_MOOD = 0.75    # 情绪定妆音和锚点的相似度下限，达不到就不用这种情绪
RIVALS = {"依绪": ["雪菜"], "雪菜": ["依绪"]}  # 选锚点时要尽量拉开的角色


def cmd_anchors(extra=5):
    """每个角色从定妆候选里重选锚点：像自己的日语原声，同时尽量不像对手角色。"""
    from mlx_audio.tts.utils import load_model

    from .speaker import SpeakerSim

    cast = load_cast(BOOK / "cast.yaml")
    refs = yaml.safe_load((WORK / "refs.yaml").read_text(encoding="utf-8"))
    sp = SpeakerSim()
    model = None
    for n in RIVALS:  # 对手角色多生成几个候选，给"拉开"留余地
        for i in range(5, 5 + extra):
            f = WORK / "refs" / f"{n}-zh-{i}.wav"
            if not f.exists():
                model = model or load_model(VOXCPM)
                a = np.concatenate([np.array(x.audio, dtype=np.float32) for x in model.generate(
                    text=refs[n]["zh_text"], ref_audio=refs[n]["jp_audio"],
                    instruct=cast["characters"][n]["style"])])
                sf.write(f, a, SR)
    jp = {n: sp.embed(r["jp_audio"]) for n, r in refs.items()}
    out = {}
    for n, r in refs.items():
        best = None
        for f in sorted((WORK / "refs").glob(f"{n}-zh-*.wav")):
            e = sp.embed(f)
            own = sp.sim(e, jp[n])
            rival = max((sp.sim(e, jp[m]) for m in RIVALS.get(n, [])), default=0.0)
            score = own - 0.5 * rival
            print(f"  {n} {f.name}：像原声 {own:.2f}，像对手 {rival:.2f}，分 {score:.2f}", flush=True)
            if best is None or score > best[0]:
                best = (score, f, own, rival)
        out[n] = {**r, "zh_audio": str(best[1]), "sim_jp": round(best[2], 3), "sim_rival": round(best[3], 3)}
        print(n, "锚点", best[1].name, flush=True)
    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    (ANCHOR_DIR / "anchors.yaml").write_text(yaml.safe_dump(out, allow_unicode=True, sort_keys=False),
                                             encoding="utf-8")


def cmd_derive(tries=6):
    """从锚点派生情绪定妆音：锚点当参考加情绪指令，CAM++ 相似度 ≥ SIM_MOOD 才采用。"""
    from mlx_audio.tts.utils import load_model

    from . import moods
    from .speaker import SpeakerSim

    cast = load_cast(BOOK / "cast.yaml")
    anchors = yaml.safe_load((ANCHOR_DIR / "anchors.yaml").read_text(encoding="utf-8"))
    sp = SpeakerSim()
    model = load_model(VOXCPM)
    out = {}
    rows = sample_rows()
    for n, anc in anchors.items():
        c = cast["characters"][n]
        e_anc = sp.embed(anc["zh_audio"])
        items = json.loads((MOOD_DIR / f"{n}-labels.json").read_text(encoding="utf-8"))
        groups = {}
        for x in items:
            groups.setdefault(x["mood"], []).append(x)
        texts = {m: moods.join_text([p["cn"] for p in g]) for m, g in
                 ((m, moods.pick(g)) for m, g in groups.items()) if g}
        need = {r["mood"] for r in rows if r["speaker"] == n and r["type"] == "台词"} - {"平常"}
        if n == cast["narrator"]:
            need.add(moods.NARRATION)
            texts[moods.NARRATION] = anc["zh_text"]
        out[n] = {"平常": {"zh_audio": anc["zh_audio"], "zh_text": anc["zh_text"],
                         "f0": anc["f0"], "rate": anc["rate"], "sim": 1.0}}
        for m in sorted(need):
            if m not in texts:
                continue
            best = None
            for i in range(tries):
                a = np.concatenate([np.array(x.audio, dtype=np.float32) for x in model.generate(
                    text=texts[m], ref_audio=anc["zh_audio"],
                    instruct=f"{c['style']}此刻：{moods.HINT[m]}。")])
                s = sp.sim(sp.embed_audio(a, SR), e_anc)
                if best is None or s > best[0]:
                    best = (s, a)
                if s >= 0.85:
                    break
            ok = best[0] >= SIM_MOOD
            print(f"{n} {m}：最高相似度 {best[0]:.2f}，{'采用' if ok else '不用，退回锚点'}", flush=True)
            if ok:
                f = ANCHOR_DIR / f"{n}-{m}.wav"
                sf.write(f, best[1], SR)
                out[n][m] = {"zh_audio": str(f), "zh_text": texts[m], "f0": qc.f0_median(best[1]) or anc["f0"],
                             "rate": anc["rate"], "sim": round(best[0], 3)}
    (ANCHOR_DIR / "refs.yaml").write_text(yaml.safe_dump(out, allow_unicode=True, sort_keys=False),
                                          encoding="utf-8")


def anchor_ref(row: dict, refs: dict, cast: dict) -> dict:
    """（说话人, 情绪）→ 参考；没有通过音色检查的情绪按回退链落到锚点。"""
    from . import moods

    n = row["speaker"]
    have = set(refs[n])
    if n == cast["narrator"] and row["type"] in ("旁白", "心理活动"):
        m = moods.NARRATION if moods.NARRATION in have else "平常"
    else:
        m = moods.resolve(row.get("mood", "平常"), cast["characters"][n]["tier"], have)
    ref = refs[n][m]
    return {**ref, "rate": min(max(ref["rate"], 3.3), 5.0), "anchor": refs[n]["平常"]["zh_audio"]}


def cmd_synth_anchor(tries=6, method="qwen-anchor"):
    """Qwen3 逐句合成，质检多一道：和锚点的 CAM++ 相似度低于 SIM_LINE 就重做。"""
    from .speaker import SpeakerSim

    cast = load_cast(BOOK / "cast.yaml")
    refs = yaml.safe_load((ANCHOR_DIR / "refs.yaml").read_text(encoding="utf-8"))
    lex = load_yaml(BOOK / "lexicon.yaml")
    sp = SpeakerSim()
    model = load_qwen()
    out = WORK / method
    out.mkdir(parents=True, exist_ok=True)
    log = {}
    for r in sample_rows():
        f = out / f"{r['idx']:04d}.wav"
        text = tts_text(r, lex)
        if f.exists() or not qc.SPOKEN.search(text):
            continue
        ref = anchor_ref(r, refs, cast)
        a, m = synth_anchor(model, sp, text, ref, tries)
        sf.write(f, a, SR)
        log[r["idx"]] = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}
        print(method, r["idx"], log[r["idx"]], flush=True)
    (out / "qc.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")


def synth_qc(model, text: str, ref: dict, tries=5, temperature=0.6, jump=1.45) -> tuple[np.ndarray, dict]:
    """Qwen3 逐句合成：音高、语速、句内变调过了就停，否则最多 tries 次取最好。"""
    cands = []
    for a in qwen_takes(model, text, ref, tries, temperature):
        cands.append(a)
        if ok(a, text, ref, jump):
            break
    return finish(cands, text, ref)


def synth_anchor(model, sp, text: str, ref: dict, tries=6, target: np.ndarray | None = None,
                 line: float = SIM_LINE) -> tuple[np.ndarray, dict]:
    """逐句多一道 CAM++ 门槛：默认和锚点比、≥SIM_LINE；原声版的锚点是日语原声，改成和原声中心比、
    过音色门槛（target、line 由调用方给）。"""
    e_anc = sp.embed(ref["anchor"]) if target is None else target
    cands = []
    for a in qwen_takes(model, text, ref, tries, 0.6):
        s = sp.sim(sp.embed_audio(a, SR), e_anc) if len(a) > SR // 2 else 1.0  # 太短的测不准
        cands.append((s, a))
        if s >= line and ok(a, text, ref, 1.45):
            break
    # 先看音色：过线的里面再按音高、语速挑；都没过线就取最像锚点的
    passed = [a for s, a in cands if s >= line]
    if passed:
        a, m = finish(passed, text, ref)
    else:
        a, m = finish([max(cands, key=lambda x: x[0])[1]], text, ref)
    m["sim"] = round(max(s for s, _ in cands), 3)
    return a, m


# ---------- 合成与质检 ----------

class TorchQwen:
    """Windows 上用官方 PyTorch 版 Qwen3-TTS（qwen-tts），参数对齐 mlx-audio 0.6.1 的带文本复刻：
    - 非流式布局（non_streaming_mode=True）：mlx 版本来就是这样；官方默认的模拟流式会越读越快；
    - 子 talker 用和主 talker 同一个温度：mlx 两者共用，官方的子 talker 默认 0.9；
    - 重复惩罚 1.5、只罚最近 64 个首码本 token：mlx 在复刻模式下强制这样，官方默认 1.05 罚全部。
      官方 generate 不往下传 logits_processor，只能包一层 talker.generate。
    - 语音编码器（编参考音、解码波形）用存储的 fp32：from_pretrained 会把它跟着主模型转成 bf16，
      mlx-audio 不转。实测 bf16 下雪菜的候选有一半音色跑偏（像锚点 0.2–0.5），fp32 降到四分之一（06 号票）。
    attention 用 sdpa，不装 flash-attn。一句的几个候选当一批生成；生成上限按句长定，
    比 ok() 的时长上限多 1 秒，防止一条失控拖住整批。
    """

    def __init__(self):
        import torch
        from qwen_tts import Qwen3TTSModel
        from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
        from transformers import LogitsProcessor, LogitsProcessorList

        class WindowPenalty(LogitsProcessor):
            def __call__(self, input_ids, scores):
                if input_ids.shape[1] == 0:
                    return scores
                ids = input_ids[:, -64:]
                s = scores.gather(1, ids)
                return scores.scatter(1, ids, torch.where(s < 0, s * 1.5, s / 1.5))

        path = hf_model(QWEN_TORCH)
        self.tts = Qwen3TTSModel.from_pretrained(str(path), device_map="cuda:0",
                                                 dtype=torch.bfloat16, attn_implementation="sdpa")
        self.tts.model.load_speech_tokenizer(Qwen3TTSTokenizer.from_pretrained(
            str(path / "speech_tokenizer"), device_map="cuda:0", dtype=torch.float32))
        talker, generate = self.tts.model.talker, self.tts.model.talker.generate

        def windowed(*a, **kw):
            return generate(*a, **{**kw, "repetition_penalty": 1.0,
                                   "logits_processor": LogitsProcessorList([WindowPenalty()])})

        talker.generate = windowed
        self.prompts = {}

    def prompt(self, ref: dict) -> list:
        """参考音编码成复刻提示并缓存。ref["xvec"] 为真时用官方的 x_vector_only_mode：只取说话人向量，
        参考码和逐字稿不进上下文（07 号票的原声向量版）；再带 spk_files 时，说话人向量取这些原声
        各自向量的平均（原声平均版）。"""
        key = (ref["zh_audio"], ref.get("xvec", False), tuple(ref.get("spk_files", ())))
        if key not in self.prompts:
            if key[2]:
                from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

                self.prompts[key] = [VoiceClonePromptItem(ref_code=None, ref_spk_embedding=self.mean_xvec(key[2]),
                                                          x_vector_only_mode=True, icl_mode=False)]
            else:
                self.prompts[key] = self.tts.create_voice_clone_prompt(
                    ref_audio=key[0], ref_text=ref["zh_text"], x_vector_only_mode=key[1])
        return self.prompts[key]

    def mean_xvec(self, files: tuple[str, ...]):
        """语音包里这些原声的说话人向量取平均，长度缩放到单句向量长度的平均值。"""
        import librosa
        import torch

        vp = VoicePak()
        sr = self.tts.model.speaker_encoder_sample_rate
        embs = [self.tts.model.extract_speaker_embedding(
            audio=librosa.resample(vp.audio(f), orig_sr=SR, target_sr=sr), sr=sr) for f in files]
        E = torch.stack([e.float() for e in embs])
        m = E.mean(0)
        return (m / m.norm() * E.norm(dim=1).mean()).to(embs[0].dtype)

    def takes(self, text: str, ref: dict, n: int, temperature: float | None) -> list[np.ndarray]:
        temp = {"temperature": temperature, "subtalker_temperature": temperature} if temperature else {}
        wavs, sr = self.tts.generate_voice_clone(
            text=[text] * n, language="Chinese", voice_clone_prompt=self.prompt(ref), non_streaming_mode=True,
            top_k=50, subtalker_top_k=50, top_p=1.0, subtalker_top_p=1.0, **temp,
            max_new_tokens=int(12.5 * (3.5 + 0.5 * len(qc.SPOKEN.findall(text)))))
        return [to48k(np.asarray(w, dtype=np.float32), sr) for w in wavs]


def load_qwen():
    if sys.platform == "win32":
        return TorchQwen()
    from mlx_audio.tts.utils import load_model

    return load_model(QWEN_BASE)


def qwen_takes(model, text: str, ref: dict, tries: int, temperature: float | None):
    """逐个给出候选（48k）。Mac 上一次生成一个，调用方 break 后就不再生成；
    Windows 上一批生成好再逐个给，调用方按同样的顺序挑，结果的分布和逐个生成一样。"""
    if isinstance(model, TorchQwen):
        yield from model.takes(text, ref, tries, temperature)
        return
    for _ in range(tries):
        res = list(model.generate(text=text, ref_audio=ref["zh_audio"], ref_text=ref["zh_text"],
                                  lang_code="chinese", temperature=temperature))
        yield to48k(np.concatenate([np.array(x.audio, dtype=np.float32) for x in res]), res[0].sample_rate)


def finish(cands: list[np.ndarray], text: str, ref: dict) -> tuple[np.ndarray, dict]:
    """从候选里挑偏差最小的，再把语速拉回目标的 0.85–1.2 倍。明细里的 pick 是挑中的是 cands 里第几个。"""
    def penalized(a):
        score, m = qc.judge(a, text, ref["f0"], ref["rate"])
        jump = qc.max_jump(a, ref["f0"])
        return score + max(0.0, jump - 1.3) * 3, {**m, "jump": round(jump, 2)}

    scored = [penalized(a) for a in cands]
    i = min(range(len(cands)), key=lambda i: scored[i][0])
    (score, m), a = scored[i], cands[i]
    if m["rate"]:
        lo, hi = ref["rate"] * 0.85, ref["rate"] * 1.2
        want = min(max(m["rate"], lo), hi)
        a = qc.stretch(a, want / m["rate"])
    return a, {"score": round(score, 3), **m, "tries": len(cands), "pick": i}


def ok(a: np.ndarray, text: str, ref: dict, jump: float | None = None, slow: float = 1.0) -> bool:
    """slow：这句有意放慢的倍数（11 号票的 duration_factor），时长上限跟着放宽。"""
    if len(a) / SR > (2.5 + 0.5 * len(qc.SPOKEN.findall(text))) * slow:
        return False
    if jump and qc.max_jump(a, ref["f0"]) > jump:
        return False
    score, _ = qc.judge(a, text, ref["f0"], ref["rate"])
    return score < 0.45  # 约等于音高偏 ±15% 或语速偏 ±45%


# 方法名 → (模型, 采样温度, 句内跳变上限, 最多试几次)
METHODS = {
    "vox": (VOXCPM, None, None, 3),
    "qwen": (QWEN_BASE, None, None, 3),
    "qwen-t06": (QWEN_BASE, 0.6, 1.45, 5),
    "qwen-mood": (QWEN_BASE, 0.6, 1.45, 5),  # 参考按（说话人, 情绪）选，见 mood_ref
}


def cmd_synth(methods: list[str]):
    refs = yaml.safe_load((WORK / "refs.yaml").read_text(encoding="utf-8"))
    mrefs = yaml.safe_load((MOOD_DIR / "refs.yaml").read_text(encoding="utf-8")) \
        if (MOOD_DIR / "refs.yaml").exists() else {}
    cast = load_cast(BOOK / "cast.yaml")
    lex = load_yaml(BOOK / "lexicon.yaml")
    rows = sample_rows()
    for method in methods:
        repo, temperature, jump, tries = METHODS[method]
        if repo == VOXCPM:
            from mlx_audio.tts.utils import load_model

            model = load_model(repo)
        else:
            model = load_qwen()
        out = WORK / method
        out.mkdir(parents=True, exist_ok=True)
        log = {}
        for r in rows:
            f = out / f"{r['idx']:04d}.wav"
            text = tts_text(r, lex)
            if f.exists() or not qc.SPOKEN.search(text):
                continue
            ref = mood_ref(r, mrefs, cast) if method == "qwen-mood" else refs[r["speaker"]]
            if repo == VOXCPM:
                cands = []
                for _ in range(tries):
                    res = list(model.generate(text=text, ref_audio=ref["zh_audio"],
                                              prompt_audio=ref["zh_audio"], prompt_text=ref["zh_text"]))
                    a = to48k(np.concatenate([np.array(x.audio, dtype=np.float32) for x in res]),
                              res[0].sample_rate)
                    cands.append(a)
                    if ok(a, text, ref, jump):
                        break
                a, m = finish(cands, text, ref)
            else:
                a, m = synth_qc(model, text, ref, tries, temperature, jump)
            sf.write(f, a, SR)
            log[r["idx"]] = m
            print(method, r["idx"], m, flush=True)
        (out / "qc.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_index():
    """把任务写成 JSON，交给 IndexTTS2 环境里的批处理脚本；回来后同样质检、挑选、变速。"""
    refs = yaml.safe_load((WORK / "refs.yaml").read_text(encoding="utf-8"))
    lex = load_yaml(BOOK / "lexicon.yaml")
    rows = sample_rows()
    raw = WORK / "index-raw"
    jobs = []
    for r in rows:
        text = tts_text(r, lex)
        if not qc.SPOKEN.search(text):
            continue
        for k in range(2):
            jobs.append({"out": str(raw / f"{r['idx']:04d}-{k}.wav"), "text": text,
                         "spk": refs[r["speaker"]]["zh_audio"], "emo_text": r["instruct"]})
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "jobs.json").write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")
    subprocess.run(["uv", "run", "--project", str(INDEX_DIR), "python",
                    str(Path(__file__).with_name("indextts_batch.py")), str(raw / "jobs.json")],
                   cwd=INDEX_DIR, check=True)
    out = WORK / "index"
    out.mkdir(parents=True, exist_ok=True)
    log = {}
    for r in rows:
        text = tts_text(r, lex)
        if not qc.SPOKEN.search(text):
            continue
        cands = []
        for k in range(2):
            a, sr = sf.read(raw / f"{r['idx']:04d}-{k}.wav", dtype="float32")
            cands.append(to48k(a.mean(axis=1) if a.ndim > 1 else a, sr))
        a, m = finish(cands, text, refs[r["speaker"]])
        sf.write(out / f"{r['idx']:04d}.wav", a, SR)
        log[r["idx"]] = m
    (out / "qc.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------- 混音与指标 ----------

def cmd_mix():
    from .mix import assemble, to_mp3

    cast = load_cast(BOOK / "cast.yaml")
    rows = sample_rows()
    lex = load_yaml(BOOK / "lexicon.yaml")
    refs = yaml.safe_load((WORK / "refs.yaml").read_text(encoding="utf-8"))
    for method in ("before", "vox", "qwen", "qwen-t06", "qwen-mood", "qwen-anchor"):
        src = OUT / "wav" / "post" / SEG if method == "before" else WORK / method
        if not src.exists():
            continue
        tmp = WORK / "mixsrc" / method
        tmp.mkdir(parents=True, exist_ok=True)
        stats = {}
        for r in rows:
            f = src / f"{r['idx']:04d}.wav"
            text = tts_text(r, lex)
            if not f.exists():  # 只有标点的句子：静音
                sf.write(tmp / f.name, np.zeros(int(0.8 * SR), dtype=np.float32), SR)
                continue
            a, _ = sf.read(f, dtype="float32")
            sf.write(tmp / f.name, a, SR)
            if r["type"] == "台词" or r["speaker"] == "春希":
                s = stats.setdefault(r["speaker"], {"f0": [], "rate": [], "jumps": 0})
                fv, rv = qc.f0_median(a), qc.rate(a, text)
                if fv:
                    s["f0"].append(fv)
                if rv:
                    s["rate"].append(rv)
                if qc.max_jump(a, refs[r["speaker"]]["f0"]) > 1.45:
                    s["jumps"] += 1
        audio = assemble(rows, tmp, cast, inner=True)
        to_mp3(audio, WORK / "mp3" / f"{method}.mp3",
               {"album": "白色相簿2 小样对比", "title": method, "artist": "AI 有声书试做"})
        print(f"== {method}  {len(audio) / SR / 60:.1f} 分钟")
        for n, s in stats.items():
            f, ra = np.array(s["f0"]), np.array(s["rate"])
            print(f"  {n}：音高 {np.median(f):.0f}Hz（{np.percentile(f, 10):.0f}–{np.percentile(f, 90):.0f}）"
                  f"  语速 {np.median(ra):.1f}（{np.percentile(ra, 10):.1f}–{np.percentile(ra, 90):.1f}）"
                  f"  句内变调 {s['jumps']} 句")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "synth":
        cmd_synth(sys.argv[2:])
    else:
        {"prepare": cmd_prepare, "index": cmd_index, "mix": cmd_mix,
         "moods-label": cmd_moods_label, "moods-build": cmd_moods_build,
         "anchors": cmd_anchors, "derive": cmd_derive, "synth-anchor": cmd_synth_anchor}[cmd]()
