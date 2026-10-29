"""14 号票：豆包语音合成 2.0 的预置音色做小样和〈2002〉，当商用对照。只发文字，不传原声。

    uv run python -m readbook.volc_passage 试音      # 每个角色几个候选音色，各说两三句
    uv run python -m readbook.volc_passage 情绪      # 情绪参数实测：同一句换不同做法
    uv run python -m readbook.volc_passage 指令+情绪 小样                # 整段：语音指令＝情绪描述＋表演提示
    uv run python -m readbook.volc_passage 指令+情绪 "2010-11-25 〈2002〉"
    uv run python -m readbook.volc_passage 锚点 指令+情绪    # 评测用的本音色锚点（定妆音的文字，不加指令）

- 音色：每个角色固定一个预置音色（PICKED，用户听试音定的），旁白和心里话也用春希的音色，靠讲述的语音指令和混音区分；
- 情绪：只用语音指令（context_texts，不计费）。实测 emotion 参数在 2.0 音色上乱写也不报错，看不出生效；
  情绪实测里用户听着"指令+情绪"（剧本情绪的描述 moods.HINT 加表演提示）最好，整段就做这一版；
  "指令"（只给表演提示）和"表现力"（再加 seed-tts-2.0-expressive）留着参数，没做整段；
- 语速：每个角色一个 speech_rate，把试音时的语速拉到定妆音的目标语速（speech_rates），不再后期变速；
- 每次合成存成 48k 干声 wav 和同名 json（参数、计费字数、耗时），参数一样就直接读，不重复计费；
- 计费字数另外逐次记在 out/volc/usage.jsonl（volc.record），`uv run python -m readbook.volc 用量` 汇总；
- 产物在 out/volc/<版本>/，要给人听的复制到 books/wa2/trials/volc/。
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from . import qc, volc
from .bakeoff import SEG, sample_rows
from .moods import HINT, NARRATION
from .synth import load_yaml, tts_text
from .trial import BOOK, OUT, read_jsonl
from .voices import SR

WORK = OUT / "volc"
TRIALS = BOOK / "trials" / "volc"
SCRIPTS = {"小样": None, "2002": "2010-11-25 〈2002〉", "2011-01-01": SEG}

# 候选音色：按 cast.yaml 的人物描述从 2.0 音色列表里挑。带 * 的在 1.0 有"多情感"版本，可能支持 emotion。
CANDIDATES = {
    "春希": [("云舟", "zh_male_m191_uranus_bigtts"), ("儒雅逸辰*", "zh_male_ruyayichen_uranus_bigtts"),
             ("清爽男大", "zh_male_qingshuangnanda_uranus_bigtts"), ("儒雅青年", "zh_male_ruyaqingnian_uranus_bigtts")],
    "雪菜": [("甜美桃子", "zh_female_tianmeitaozi_uranus_bigtts"), ("邻家女孩", "zh_female_linjianvhai_uranus_bigtts"),
             ("柔美女友*", "zh_female_roumeinvyou_uranus_bigtts"), ("小何", "zh_female_xiaohe_uranus_bigtts")],
    "依绪": [("爽快思思*", "zh_female_shuangkuaisisi_uranus_bigtts"), ("直率英子", "zh_female_zhishuaiyingzi_uranus_bigtts"),
             ("开朗姐姐", "zh_female_kailangjiejie_uranus_bigtts"), ("Vivi", "zh_female_vv_uranus_bigtts")],
    "武也": [("阳光青年*", "zh_male_yangguangqingnian_uranus_bigtts"), ("活力小哥", "zh_male_huolixiaoge_uranus_bigtts"),
             ("开朗学长", "zh_male_kailangxuezhang_uranus_bigtts"), ("小天", "zh_male_taocheng_uranus_bigtts")],
}
# 试音句：(段, idx)。春希多一句旁白；武也在小样里只有一句话，另取 2011-01-01 的一句。
AUDITION = {
    "春希": [("2002", 22), ("小样", 77), ("小样", 74)],
    "雪菜": [("2002", 14), ("小样", 80)],
    "依绪": [("小样", 62), ("小样", 58)],
    "武也": [("小样", 3), ("2011-01-01", 147)],
}


def rows_of(seg: str) -> dict[int, dict]:
    rows = sample_rows() if seg == "小样" else read_jsonl(BOOK / "script" / f"{SCRIPTS[seg]}.jsonl")
    return {r["idx"]: r for r in rows}


def narration(row: dict, cast: dict) -> bool:
    return row["speaker"] == cast["narrator"] and row["type"] in ("旁白", "心理活动")


def take(path: Path, text: str, speaker: str, purpose: str, **kw) -> tuple[np.ndarray, dict]:
    """合成一句，存成 48k 干声 wav 和同名 json；参数一样就直接读，不重复计费。"""
    params = {"text": text, "speaker": speaker, **{k: v for k, v in kw.items() if v}}
    js = path.with_suffix(".json")
    if path.exists() and js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        if rec["params"] == params:
            return sf.read(path, dtype="float32")[0], rec
    pcm, end = volc.synth(text, speaker, fmt="pcm", sample_rate=SR, purpose=purpose, **kw)
    a = np.frombuffer(pcm, "<i2").astype(np.float32) / 32768
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, a, SR)
    rec = {"params": params, "billed": end.get("usage", {}).get("text_words"), "secs": end["secs"],
           "dur": round(len(a) / SR, 2), "logid": end["logid"]}
    js.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    return a, rec


def trim(a: np.ndarray) -> np.ndarray:
    """切掉首尾静音（阈值同 qc.speech_seconds），两头各留 0.1 秒。"""
    idx = np.flatnonzero(np.abs(a) > 0.02 * (np.abs(a).max() + 1e-9))
    if not len(idx):
        return a
    pad = int(0.1 * SR)
    return a[max(idx[0] - pad, 0): idx[-1] + pad]


def stamp(t: float) -> str:
    m, d = divmod(round(t * 10), 600)
    return f"{m}:{d // 10:02d}.{d % 10}"


def real_centers(names: list[str]) -> tuple[object, dict]:
    """CAM++ 和每个角色日语原声的中心（只在本机算，原声不出本机）。"""
    from .evaluate import real_embeddings, voice_gate
    from .scriptize import load_cast
    from .speaker import SpeakerSim

    sp = SpeakerSim()
    real = real_embeddings(sp, load_cast(BOOK / "cast.yaml"), names)
    return sp, {n: voice_gate(E) for n, E in real.items()}


# ---------- 试音 ----------

def audition():
    from .mix import norm, to_mp3
    from .scriptize import load_cast

    cast = load_cast(BOOK / "cast.yaml")
    lex = load_yaml(BOOK / "lexicon.yaml")
    targets = {n: v["定妆音"]["平常"] for n, v in load_yaml(BOOK / "voices" / "voices.yaml").items()}
    segs = {s: rows_of(s) for s in SCRIPTS}
    sp, gates = real_centers(list(AUDITION))
    out = TRIALS / "试音"
    out.mkdir(parents=True, exist_ok=True)
    md = ["# 14 号票试音：豆包语音合成 2.0 预置音色", "",
          "每个角色几个候选音色，各说同样的两三句。语音指令（context_texts）给的是剧本的表演提示，"
          "春希的旁白给讲述提示；没调语速、音量，没用 emotion 参数。",
          "目标音高、语速取混合版定妆音（平常）。像原声＝和本人日语原声中心的 CAM++ 相似度，预置音色不是本人，"
          "只作参考（本人原声的音色门槛：春希 0.45、雪菜 0.40、依绪 0.59、武也 0.55）。", ""]
    summary = ["## 汇总", "", "| 角色 | 候选 | 音高 Hz 中位（目标） | 语速 字/秒 中位（目标） | 像原声 中位 | 每字耗时 秒 |",
               "|---|---|---|---|---|---|"]
    for name, lines in AUDITION.items():
        md += [f"## {name}（`试音-{name}.mp3`）", "",
               "| 时间 | 候选 | 句 | 提示 | 音高 Hz | 语速 字/秒 | 像原声 | 耗时 秒 | 计费字数 |",
               "|---|---|---|---|---|---|---|---|---|"]
        chunks, pos = [], 0.0
        for label, speaker in CANDIDATES[name]:
            f0s, rates, sims, secs, chars = [], [], [], 0.0, 0
            for k, (seg, i) in enumerate(lines):
                row = segs[seg][i]
                text = tts_text(row, lex)
                hint = HINT[NARRATION] if narration(row, cast) else row.get("instruct")
                a, rec = take(WORK / "试音" / name / f"{speaker}-{seg}-{i:04d}.wav", text, speaker,
                              f"14号票:试音 {name}", context=hint)
                a = trim(a)
                f0, rate = qc.f0_median(a), qc.rate(a, text)
                sim = float(sp.embed_audio(a, SR) @ gates[name][0])
                f0s.append(f0), rates.append(rate), sims.append(sim)
                secs += rec["secs"]
                chars += len(qc.SPOKEN.findall(text))
                gap = 1.5 if k == 0 and chunks else 0.6 if chunks else 0.0
                chunks.append(np.zeros(int(gap * SR), np.float32))
                pos += gap
                md.append(f"| {stamp(pos)} | {label} | {seg} #{i} | {hint} | {f0 or 0:.0f} "
                          f"| {f'{rate:.2f}' if rate else '—'} | {sim:.2f} | {rec['secs']:.1f} | {rec['billed']} |")
                chunks.append(norm(a))
                pos += len(a) / SR
            ok = [x for x in rates if x]
            summary.append(f"| {name} | {label} `{speaker}` | {np.median([x for x in f0s if x]):.0f}"
                           f"（{targets[name]['f0']:.0f}） | {np.median(ok):.2f}（{targets[name]['rate']:.2f}） "
                           f"| {np.median(sims):.2f} | {secs / chars:.3f} |")
        md.append("")
        audio = np.concatenate(chunks)
        to_mp3(audio * 0.97 / max(np.abs(audio).max(), 0.97), out / f"试音-{name}.mp3",
               {"album": "白色相簿2 剧本全集（试做）", "title": f"14 号票试音：{name}", "artist": "豆包语音合成 2.0"})
    u = volc.usage()
    md += summary + ["", f"计费：到目前为止一共 {u['total']} 字（免费额度 {volc.FREE} 字，还剩 {u['left']}）。", ""]
    (out / "试音.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(out / "试音.md")


# ---------- 情绪参数实测 ----------

# 试音后的推荐组合，情绪实测用的是它们。带 * 的音色 1.0 版有多情感，试 emotion 参数。
RECOMMENDED = {"春希": "zh_male_qingshuangnanda_uranus_bigtts", "雪菜": "zh_female_roumeinvyou_uranus_bigtts",
          "依绪": "zh_female_zhishuaiyingzi_uranus_bigtts", "武也": "zh_male_yangguangqingnian_uranus_bigtts"}
MULTI_EMOTION = {"zh_female_roumeinvyou_uranus_bigtts", "zh_male_yangguangqingnian_uranus_bigtts",
                 "zh_male_ruyayichen_uranus_bigtts", "zh_female_shuangkuaisisi_uranus_bigtts"}
# 用户听完试音定的，整段用它们。
PICKED = {"春希": "zh_male_ruyayichen_uranus_bigtts", "雪菜": "zh_female_tianmeitaozi_uranus_bigtts",
          "依绪": "zh_female_zhishuaiyingzi_uranus_bigtts", "武也": "zh_male_taocheng_uranus_bigtts"}
# 剧本情绪 → audio_params.emotion（1.0 多情感音色的取值）
EMO = {"开心": "happy", "兴奋": "excited", "生气": "angry", "难过": "sad", "哭腔": "sad", "害羞": "shy",
       "冷淡": "coldness", "惊讶": "surprised", "轻声": "ASMR", "激动": "excited", "低落": "depressed"}
# 剧本情绪 → (speech_rate, loudness_rate)
PACE = {"轻声": (-10, -30), "难过": (-10, -10), "哭腔": (-10, 0), "兴奋": (10, 20), "生气": (5, 20),
        "惊讶": (0, 10), "害羞": (-5, -20), "冷淡": (-5, -10), "开心": (5, 0), "激动": (10, 20), "低落": (-10, -10)}
# 情绪 → emotion2vec 的类别（能对上的才算）
E2V = {"开心": "开心", "兴奋": "开心", "生气": "生气", "难过": "难过", "哭腔": "难过", "惊讶": "吃惊"}
EMOTEST = [("雪菜", "小样", 80), ("雪菜", "小样", 75), ("雪菜", "2002", 19), ("雪菜", "2002", 12),
           ("雪菜", "2002", 4), ("雪菜", "小样", 59), ("依绪", "小样", 58), ("依绪", "小样", 12),
           ("春希", "2002", 63), ("春希", "小样", 83), ("武也", "2011-01-01", 141)]
EXPRESSIVE = "seed-tts-2.0-expressive"


def variants(row: dict, speaker: str) -> dict[str, dict]:
    """同一句的几种做法 → synth 的参数。"""
    mood, ins = row.get("mood", "平常"), row.get("instruct")
    out = {"平": {}, "指令": {"context": ins}, "指令·重做": {"context": ins},
           "指令+情绪": {"context": f"{HINT[mood]}，{ins}"},
           "表现力": {"context": ins, "model": EXPRESSIVE}}
    if speaker in MULTI_EMOTION and mood in EMO:
        out["情感参数"] = {"emotion": EMO[mood]}
        out["情感参数+指令"] = {"emotion": EMO[mood], "context": ins}
    if mood in PACE:
        out["指令+语速音量"] = {"context": ins, "speech_rate": PACE[mood][0], "loudness_rate": PACE[mood][1]}
    return out


def emotest():
    """推荐音色上，同一句换不同做法：识别情绪、音高、语速、响度、字错率、和本音色"平"那版像不像。"""
    import torch
    import torchaudio

    from .asrcheck import TorchWhisper, cer
    from .evaluate import Emotion
    from .mix import norm, to_mp3
    from .speaker import SpeakerSim

    lex = load_yaml(BOOK / "lexicon.yaml")
    segs = {s: rows_of(s) for s in SCRIPTS}
    takes = []
    for name, seg, i in EMOTEST:
        row, speaker = segs[seg][i], RECOMMENDED[name]
        text = tts_text(row, lex)
        for v, kw in variants(row, speaker).items():
            try:
                a, rec = take(WORK / "情绪" / f"{seg}-{i:04d}-{v}.wav", text, speaker, f"14号票:情绪 {v}", **kw)
            except Exception as e:  # 不支持的参数组合会报错（HTTP 4xx 或错误码），记下来
                print(name, seg, i, v, e, flush=True)
                takes.append((name, seg, i, v, kw, None, {"error": str(e)}))
                continue
            takes.append((name, seg, i, v, kw, trim(a), rec))
    sp, wh, emo = SpeakerSim(), TorchWhisper(), Emotion()
    plain = {}
    for name, seg, i, v, kw, a, rec in takes:
        if v == "平" and a is not None:
            plain[(seg, i)] = sp.embed_audio(a, SR)
    md = ["# 14 号票：情绪参数实测", "",
          f"推荐音色上，同一句换几种做法。音色：" + "，".join(f"{n} `{s}`" for n, s in RECOMMENDED.items()) + "。",
          "- 平：什么都不加；指令：语音指令（context_texts）给剧本的表演提示；指令·重做：同样的参数再请求一次，看随机性；",
          "- 指令+情绪：语音指令前面加上情绪的描述（moods.HINT）；表现力：指令 + model=seed-tts-2.0-expressive；",
          "- 情感参数：audio_params.emotion（只试 1.0 版有多情感的音色）；指令+语速音量：指令 + speech_rate / loudness_rate。",
          "- 识别情绪：emotion2vec+ 的第一名；目标类概率：剧本情绪能对上 emotion2vec 类别时，那一类的概率；"
          "像平：和同一句「平」那版的 CAM++ 相似度（音色有没有被带偏）；响度：合成原样的 RMS dBFS。", "",
          "| 时间 | 谁 | 句 | 情绪 | 做法 | 识别情绪 | 目标类概率 | 音高 Hz | 语速 | 响度 dB | 像平 | 字错率 | 识别 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    chunks, pos, last = [], 0.0, None
    for name, seg, i, v, kw, a, rec in takes:
        row = segs[seg][i]
        text = tts_text(row, lex)
        if a is None:
            md.append(f"| — | {name} | {seg} #{i} | {row['mood']} | {v} | 报错：{rec['error'][:40]} |||||||| ")
            continue
        w = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(a)), SR, 16000).numpy()
        res = emo.model.generate(w, granularity="utterance", extract_embedding=False, disable_pbar=True)[0]
        scores = {l.split("/")[0]: float(s) for l, s in zip(res["labels"], res["scores"])}
        top = max(scores, key=scores.get)
        want = E2V.get(row["mood"])
        hyp = wh.transcribe_audio(a, SR)
        f0, rate = qc.f0_median(a), qc.rate(a, text)
        db = 20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-9)
        like = float(sp.embed_audio(a, SR) @ plain[(seg, i)]) if (seg, i) in plain and v != "平" else None
        gap = 1.5 if (seg, i) != last and chunks else 0.6 if chunks else 0.0
        chunks.append(np.zeros(int(gap * SR), np.float32))
        pos += gap
        last = (seg, i)
        md.append(f"| {stamp(pos)} | {name} | {seg} #{i} | {row['mood']}·{row.get('instruct')} | {v} | {top} {scores[top]:.2f} "
                  f"| {f'{scores[want]:.2f}' if want else '—'} | {f0 or 0:.0f} | {f'{rate:.2f}' if rate else '—'} "
                  f"| {db:.1f} | {f'{like:.2f}' if like is not None else '—'} | {cer(text, hyp):.2f} | {hyp} |")
        chunks.append(norm(a))
        pos += len(a) / SR
    out = TRIALS / "情绪"
    out.mkdir(parents=True, exist_ok=True)
    audio = np.concatenate(chunks)
    to_mp3(audio * 0.97 / max(np.abs(audio).max(), 0.97), out / "情绪实测.mp3",
           {"album": "白色相簿2 剧本全集（试做）", "title": "14 号票：情绪参数实测", "artist": "豆包语音合成 2.0"})
    u = volc.usage()
    md += ["", f"计费：到目前为止一共 {u['total']} 字（还剩 {u['left']}）。", ""]
    (out / "情绪实测.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(out / "情绪实测.md")


# ---------- 整段 ----------

# 版本 → (语音指令前面加不加情绪描述, req_params.model)
VERSIONS = {"指令+情绪": (True, None), "指令": (False, None), "表现力": (False, EXPRESSIVE)}


def round5(x: float) -> int:
    return int(5 * round(x / 5))


def speech_rates(cast_voices: dict[str, str], targets: dict[str, dict]) -> dict[str, int]:
    """每个角色一个 speech_rate：把这个音色试音时（带语音指令）的语速中位数拉到定妆音的目标语速。
    语速参数是百分比（-50～100），取到 5 的倍数。"""
    lex = load_yaml(BOOK / "lexicon.yaml")
    segs = {s: rows_of(s) for s in SCRIPTS}
    out = {}
    for name, speaker in cast_voices.items():
        rates = []
        for seg, i in AUDITION[name]:
            f = WORK / "试音" / name / f"{speaker}-{seg}-{i:04d}.wav"
            r = qc.rate(trim(sf.read(f, dtype="float32")[0]), tts_text(segs[seg][i], lex))
            if r:
                rates.append(r)
        out[name] = round5(100 * (targets[name]["rate"] / np.median(rates) - 1))
    return out


def context(row: dict, cast: dict, mood: bool) -> str | None:
    """语音指令：剧本的表演提示，mood 时前面加上情绪的描述（同情绪实测的"指令+情绪"）；
    叙述者情绪平常的旁白和心里话（讲述模式）前面加的是讲述的提示。"""
    ins = row.get("instruct")
    if narration(row, cast) and row.get("mood", "平常") == "平常":
        return f"{HINT[NARRATION]}；这一句：{ins}" if ins else HINT[NARRATION]
    if not mood:
        return ins
    hint = HINT[row.get("mood", "平常")]
    return f"{hint}，{ins}" if ins else hint


def anchors(version: str) -> dict[str, Path]:
    """评测用的本音色锚点：每个音色说一遍自己定妆音的文字（平常，不加指令），只用来量"前后稳不稳"。"""
    voices = load_yaml(BOOK / "voices" / "voices.yaml")
    out = {}
    for name, speaker in PICKED.items():
        text = voices[name]["定妆音"]["平常"]["text"]
        a, _ = take(WORK / "锚点" / version / f"{speaker}.wav", text, speaker, f"14号票:锚点 {name}",
                    model=VERSIONS[version][1])
        out[name] = TRIALS / "锚点" / version / f"{name}.wav"
        out[name].parent.mkdir(parents=True, exist_ok=True)
        sf.write(out[name], trim(a), SR)
    return out


def build(version: str, name: str):
    from .passage import load_rows, mix
    from .scriptize import load_cast

    rows = load_rows(name, None, None, None)
    cast = load_cast(BOOK / "cast.yaml")
    lex = load_yaml(BOOK / "lexicon.yaml")
    targets = {n: v["定妆音"]["平常"] for n, v in load_yaml(BOOK / "voices" / "voices.yaml").items()}
    mood, model = VERSIONS[version]
    rates = speech_rates(PICKED, targets)
    print("speech_rate", rates, flush=True)
    out = WORK / version / name
    wav = out / "wav"
    wav.mkdir(parents=True, exist_ok=True)
    log = {}
    for r in rows:
        f = wav / f"{r['idx']:04d}.wav"
        text = tts_text(r, lex)
        if not qc.SPOKEN.search(text):
            sf.write(f, np.zeros(int(0.8 * SR), dtype=np.float32), SR)
            continue
        n = r["speaker"]
        ctx = context(r, cast, mood)
        a, rec = take(out / "raw" / f"{r['idx']:04d}.wav", text, PICKED[n], f"14号票:{version} {name}",
                      context=ctx, speech_rate=rates[n], model=model)
        a = trim(a)
        sf.write(f, a, SR)
        log[str(r["idx"])] = {"speaker": PICKED[n], "context": ctx, "speech_rate": rates[n], "billed": rec["billed"],
                              "secs": rec["secs"], "dur": round(len(a) / SR, 2), "chars": len(qc.SPOKEN.findall(text)),
                              "f0": qc.f0_median(a), "rate": qc.rate(a, text)}
        print(r["idx"], log[str(r["idx"])], flush=True)
    (out / "qc.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    label = f"{name}-豆包2.0-{version}"
    mix(label, rows, cast, wav, out)
    TRIALS.mkdir(parents=True, exist_ok=True)
    (TRIALS / f"{label}.mp3").write_bytes((out / f"{label}.mp3").read_bytes())
    (TRIALS / f"{label}-对照剧本.md").write_text((out / f"{label} 对照剧本.md").read_text(encoding="utf-8"),
                                              encoding="utf-8", newline="\n")
    print(TRIALS / f"{label}.mp3")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1] == "试音":
        audition()
    elif sys.argv[1] == "情绪":
        emotest()
    elif sys.argv[1] == "锚点":
        print(anchors(sys.argv[2]))
    else:
        build(sys.argv[1], sys.argv[2])
