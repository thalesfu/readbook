"""19 号票：修转 48 kHz 的镜像，试高频处理。

IndexTTS-2.5 输出 22.05 kHz，以前用线性插值转 48 kHz（没有抗镜像滤波），11 kHz 以上翻折出一层镜像，
耳机里听着像电子颤音；现在 voices.to48k 改用 soxr。不重新合成：index_passage repick 拿现成候选重新挑选，
成品在 out/index/<版本>-重采样/<片段>/wav/。高频处理放在混音（mix.HIGHS），这里做试听材料。

    uv run python -m readbook.resample 镜像            # 修前修后各频段的能量，写 books/wa2/trials/resample/镜像.md
    uv run python -m readbook.resample ab              # 春希颤音 A/B：out/resample/ab/高频AB.wav 和时间表
    uv run python -m readbook.resample 混音 <档> [目录]  # 小样和〈2002〉按 mix.HIGHS 的这一档重新混音，默认放 trials/resample/
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import welch

from .evaluate import decode
from .index_passage import WORK
from .mix import HIGHS, process, tame
from .passage import load_rows, remix, stamp
from .scriptize import load_cast
from .trial import BOOK, OUT
from .voices import SR

TRIAL = BOOK / "trials" / "resample"
AB = OUT / "resample" / "ab"
# 片段 → 候选所在的版本（〈2002〉里没有武也，新规和全员一样）
SOURCE = {"小样": "2.5-原声平均锚点新规", "2010-11-25 〈2002〉": "2.5-原声平均锚点全员"}
SHORT = {"小样": "小样", "2010-11-25 〈2002〉": "〈2002〉"}
BANDS = [(4000, 8000), (8000, 10000), (10000, 11000), (11000, 12000), (12000, 16000)]
# A/B 的句子：17 号票用户标过颤音的春希 7 句，再加雪菜、依绪各一句听闷不闷
AB_LINES = [("小样", 1), ("小样", 4), ("小样", 7), ("小样", 13),
            ("2010-11-25 〈2002〉", 22), ("2010-11-25 〈2002〉", 25), ("2010-11-25 〈2002〉", 38),
            ("小样", 63), ("小样", 66)]


def wav_dir(name: str, fixed: bool) -> Path:
    v = SOURCE[name]
    return WORK / (f"{v}-重采样" if fixed else v) / name / "wav"


def bands(a: np.ndarray) -> list[float]:
    """各频段相对 0.1–4 kHz 的平均功率谱密度，dB。"""
    f, p = welch(a, SR, nperseg=4096)
    ref = p[(f >= 100) & (f < 4000)].mean()
    return [10 * np.log10(p[(f >= lo) & (f < hi)].mean() / ref + 1e-30) for lo, hi in BANDS]


def cmd_mirror():
    head = " | ".join(f"{lo / 1000:g}–{hi / 1000:g} kHz" for lo, hi in BANDS)
    lines = ["# 19 号票：转 48 kHz 的镜像，修前修后", "",
             "各频段平均功率谱密度相对 0.1–4 kHz 的 dB。IndexTTS-2.5 原始输出是 22.05 kHz，真实内容只到 11.025 kHz，"
             "这以上的能量都是转 48 kHz 时翻折上去的镜像。干声＝挑中的逐句 wav 首尾相接；MP3＝混好的成品"
             "（LAME 160k，16 kHz 以上本来就被它截掉，所以只量到 16 kHz）。", "",
             f"| 片段 | 音频 | 转 48 kHz | {head} |", "|---|---|---|" + "---|" * len(BANDS)]
    for name, v in SOURCE.items():
        rows = {r["idx"] for r in load_rows(name, None, None, None)}
        for fixed in (False, True):
            d = wav_dir(name, fixed)
            a = np.concatenate([sf.read(d / f"{i:04d}.wav", dtype="float32")[0] for i in sorted(rows)])
            how = "soxr" if fixed else "线性插值"
            lines.append(f"| {SHORT[name]} | 干声 | {how} | " + " | ".join(f"{x:.1f}" for x in bands(a)) + " |")
        for fixed in (False, True):
            mp3 = next(f for f in sorted(wav_dir(name, fixed).parent.glob("*.mp3")) if "无电话效果" not in f.name)
            how = "soxr" if fixed else "线性插值"
            lines.append(f"| {SHORT[name]} | MP3 | {how} | " + " | ".join(f"{x:.1f}" for x in bands(decode(mp3))) + " |")
    TRIAL.mkdir(parents=True, exist_ok=True)
    md = TRIAL / "镜像.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(md)


def cmd_ab():
    """每句连着放几版：A 修前的成品（线性插值，不另处理），B–F 修后的成品配 mix.HIGHS 的各档。
    走真正的混音链（mix.process：响度拉平、台词的房间混响、定位），只是不拼成整段。"""
    cast = load_cast(BOOK / "cast.yaml")
    versions = [("A", "修前（线性插值，有镜像）", False, "不处理")] + [
        (chr(ord("B") + k), f"修后 · {h}", True, h) for k, h in enumerate(HIGHS)]
    rows = {n: {r["idx"]: r for r in load_rows(n, None, None, None)} for n in SOURCE}
    chunks, pos, grid = [], 0, []
    for n, i in AB_LINES:
        r = rows[n][i]
        p = cast["characters"].get(r["speaker"], {}).get("pan", 0.0)
        cells = []
        for _, _, fixed, h in versions:
            a, _ = sf.read(wav_dir(n, fixed) / f"{i:04d}.wav", dtype="float32")
            st = process(a, r, p, inner=True, highs=h).astype(np.float32)
            cells.append(stamp(pos / SR))
            chunks += [st, np.zeros((int(0.8 * SR), 2), np.float32)]
            pos += len(st) + int(0.8 * SR)
        chunks.append(np.zeros((int(1.5 * SR), 2), np.float32))
        pos += int(1.5 * SR)
        kind = {"旁白": "旁白", "台词": "台词", "心理活动": "心里话"}[r["type"]]
        text = r["text"].strip("“”")
        grid.append(f"| {SHORT[n]} {i} 号 · {r['speaker']}{kind}「{text[:14]}{'…' if len(text) > 14 else ''}」 | "
                    + " | ".join(cells) + " |")
    out = np.concatenate(chunks)
    out *= min(1.0, 0.97 / float(np.abs(out).max()))
    AB.mkdir(parents=True, exist_ok=True)
    sf.write(AB / "高频AB.wav", out, SR, subtype="PCM_16")

    def curve(h):
        spec = HIGHS[h]
        if spec is None:
            return "不另处理"
        how = (f"一阶低通，{spec[1] / 1000:g} kHz 起每倍频程 −6 dB" if spec[0] == "lp1"
               else f"高架，{spec[1] / 1000:g} kHz 起压低")
        return how + "（" + "、".join(f"{f / 1000:g} kHz {round(g) or 0}".replace("-", "−")
                                       for f, g in response(h, (2000, 4000, 6000, 10000))) + " dB）"

    lines = ["# 高频处理 A/B（19 号票）", "",
             f"`高频AB.wav`，{len(out) / SR / 60:.1f} 分钟，48 kHz 无损。每句连着放 {len(versions)} 版，版与版之间停 0.8 秒，"
             "句与句之间停 2.3 秒。走的是正式的混音链（响度拉平、台词加房间混响、立体声定位），每版响度一样。", "",
             "| 版 | 做了什么 |", "|---|---|"]
    lines += [f"| {k} | {label}：{curve(h)} |" for k, label, _, h in versions]
    lines += ["", "请戴 AirPods Pro 3 听，在每个时间后面标颤音：无 / 轻 / 中 / 重；最后一行标每版整体闷不闷、"
              "读音清不清楚。雪菜、依绪那两句主要听闷不闷。", "",
              "| 句子 | " + " | ".join(k for k, *_ in versions) + " |", "|---|" + "---|" * len(versions)]
    lines += grid + ["| 闷不闷 / 读音清不清楚 | " + " | ".join("" for _ in versions) + " |"]
    (AB / "高频AB.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(AB / "高频AB.wav", f"{len(out) / SR / 60:.1f} 分钟")


def response(highs: str, freqs) -> list[tuple[float, float]]:
    """mix.tame 这一档在各频率上的增益（dB）。"""
    imp = np.zeros(1 << 16)
    imp[0] = 1
    H = np.abs(np.fft.rfft(tame(imp, highs)))
    f = np.fft.rfftfreq(len(imp), 1 / SR)
    return [(x, 20 * np.log10(H[np.argmin(np.abs(f - x))])) for x in freqs]


def cmd_mix(highs: str, out: str | None = None):
    assert highs in HIGHS, f"档只能是 {list(HIGHS)}"
    dst = Path(out) if out else TRIAL
    for name, v in SOURCE.items():
        remix(name, wav_dir(name, True), dst, f"{name}-IndexTTS-{v}-重采样-{highs}", highs=highs)


if __name__ == "__main__":
    {"镜像": cmd_mirror, "ab": cmd_ab, "混音": cmd_mix}[sys.argv[1]](*sys.argv[2:])
