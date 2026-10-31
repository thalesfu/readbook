"""合成质检：音高、语速，以及不改音高的变速。

音高用自相关粗估（有声帧的中位数），只用来发现"春希飘进女声音区"这种大偏差，不追求精确。
"""

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 48000
# 念出来的字，字数都按它数（语速、时长上限、生成上限）。读音表的拼音标注 <字|拼音声调> 只算那个字：
# 后面紧跟着「不含 < > | 的一串再接 >」的，是标注里的拼音，不算
SPOKEN = re.compile(r"\w(?![^<>|]*>)", re.U)


def f0_median(a: np.ndarray, sr: int = SR) -> float | None:
    a = a / (np.abs(a).max() + 1e-9)
    a, sr = a[::3], sr // 3
    n = int(0.04 * sr)
    lo, hi = int(sr / 400), int(sr / 70)
    fs = []
    for i in range(0, len(a) - n, n // 2):
        x = a[i:i + n] - a[i:i + n].mean()
        if np.sqrt((x ** 2).mean()) < 0.02:
            continue
        r = np.correlate(x, x, "full")[n - 1:]
        r /= r[0] + 1e-9
        k = lo + int(np.argmax(r[lo:hi]))
        if r[k] > 0.5:
            fs.append(sr / k)
    return float(np.median(fs)) if len(fs) > 5 else None


def speech_seconds(a: np.ndarray, sr: int = SR) -> float:
    """去掉首尾静音后的时长。"""
    env = np.abs(a) > 0.02 * (np.abs(a).max() + 1e-9)
    idx = np.flatnonzero(env)
    return (idx[-1] - idx[0]) / sr if len(idx) else 0.0


def rate(a: np.ndarray, text: str, sr: int = SR) -> float | None:
    """每秒字数；不到 4 个字的句子不算。"""
    n = len(SPOKEN.findall(text))
    s = speech_seconds(a, sr)
    return n / s if n >= 4 and s > 0 else None


def stretch(a: np.ndarray, factor: float, sr: int = SR) -> np.ndarray:
    """变速不变调；factor > 1 变快。"""
    if abs(factor - 1) < 0.02:
        return a
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "a.wav", Path(d) / "b.wav"
        sf.write(src, a, sr)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                        "-filter:a", f"atempo={factor:.4f}", str(dst)], check=True)
        out, _ = sf.read(dst, dtype="float32")
    return out


def max_jump(a: np.ndarray, target_f0: float, sr: int = SR) -> float:
    """句内每半秒一段估音高，返回最高一段与目标的比值；用来抓"说着说着变调"。

    低于目标一半的估计多半是倍频误判，不算。
    """
    seg = int(0.5 * sr)
    fs = [f0_median(a[i:i + seg], sr) for i in range(0, max(len(a) - seg // 2, 1), seg)]
    fs = [f for f in fs if f and f > target_f0 * 0.5]
    return max(fs) / target_f0 if fs else 1.0


def jitter(a: np.ndarray, sr: int = SR) -> float | None:
    """音高抖动（音分），抓合成的颤音（16 号票对比文档的算法）：

    pyin 每 10 毫秒估一个音高换成音分，只看连续有声不短于 0.3 秒的段，减掉 0.3 秒的滑动平均（去掉语调起伏），
    丢掉两头各半个窗口和 |残差| ≥ 300 音分的帧（多半是倍频误判），剩下的残差取稳健标准差（1.4826×MAD）。
    会把正常的语调起伏也算进一部分，只拿来比同一个人的原声和合成。测不出（没有够长的有声段）返回 None。
    17 号票试听：它和耳机里听到的电子颤音对不上（三个春希锚点上正好排反），不再用来挑锚点。
    """
    import librosa

    a = a.mean(axis=1) if a.ndim > 1 else a
    a = librosa.resample(a, orig_sr=sr, target_sr=16000)
    f0, voiced, _ = librosa.pyin(a, fmin=65, fmax=500, sr=16000, frame_length=1024, hop_length=160)
    k = 30
    res, i, n = [], 0, len(f0)
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        while j < n and voiced[j]:
            j += 1
        if j - i >= k:
            c = 1200 * np.log2(f0[i:j] / 100)
            r = (c - np.convolve(c, np.ones(k) / k, "same"))[k // 2: -k // 2]
            r = r[np.abs(r) < 300]
            if len(r) >= 16:
                res.append(r)
        i = j
    if not res:
        return None
    r = np.concatenate(res)
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def judge(a: np.ndarray, text: str, target_f0: float, target_rate: float) -> tuple[float, dict]:
    """偏差分数（越小越好）和明细。音高按对数偏差，语速按比例偏差。"""
    f, r = f0_median(a), rate(a, text)
    df = abs(np.log(f / target_f0)) if f else 0.0
    dr = abs(np.log(r / target_rate)) if r else 0.0
    return df * 2 + dr, {"f0": f, "rate": r}
