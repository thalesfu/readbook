"""混音：逐句后期 → 立体声定位 → 拼接 → MP3。

后期链（都在 48kHz 单声道上做，最后再摆到立体声里）：
- 每句先把响度拉到同一水平；
- 心理活动（inner=True 时）：压低音量、去低频、轻微近讲提亮，再加短混响；
- 电话：300–3400Hz 带通加轻度削波；
- 回忆：去掉一点高频、压低音量，再加稍长一些的混响；
- 书信短信：不加效果；
- 台词（没有别的效果时）加一点很短的房间混响，旁白保持干声并稍压低一点，
  让"在场景里说话"和"给听众讲述"听得出区别。
混响用两条去相关的衰减噪声做脉冲响应，左右声道不同，听起来是立体声。
"""

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt

SR = 48000
TARGET_RMS = 10 ** (-20 / 20)
GAP = {"same": 0.25, "change": 0.4, "narration": 0.5}


def norm(a: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(a ** 2))) or 1.0
    return a * (TARGET_RMS / rms)


def db(a: np.ndarray, gain: float) -> np.ndarray:
    return a * 10 ** (gain / 20)


def filt(a: np.ndarray, kind: str, freq) -> np.ndarray:
    return sosfilt(butter(4, freq, kind, fs=SR, output="sos"), a)


def reverb_ir(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)  # 到末尾衰减 60dB
    ir = filt(ir, "lowpass", 5000)
    return ir / np.sqrt(np.sum(ir ** 2))


IR = {k: (reverb_ir(s, 1), reverb_ir(s, 2)) for k, s in {"inner": 0.35, "memory": 0.8, "room": 0.25}.items()}


def wet(a: np.ndarray, kind: str, mix: float) -> np.ndarray:
    """返回 (n, 2) 的立体声：干声居中加左右不同的混响。"""
    l, r = (fftconvolve(a, ir)[: len(a) + len(ir) // 2] for ir in IR[kind])
    dry = np.pad(a, (0, len(l) - len(a)))
    return np.stack([dry + mix * l, dry + mix * r], axis=1)


def pan(a: np.ndarray, p: float) -> np.ndarray:
    """等功率定位，p ∈ [-1, 1]。"""
    theta = (p + 1) * np.pi / 4
    return np.stack([a * np.cos(theta), a * np.sin(theta)], axis=1) * np.sqrt(2)


def process(a: np.ndarray, row: dict, p: float, inner: bool) -> np.ndarray:
    a = norm(a)
    effect = row["effect"]
    if row["type"] == "心理活动" and inner:
        a = db(filt(a, "highpass", 120), -4)
        st = wet(a, "inner", 0.18)
        return st  # 心理活动一律居中
    if effect == "电话":
        a = np.tanh(filt(a, "bandpass", (300, 3400)) * 3) / 3
        a = norm(a) * 0.8
    if effect == "回忆":
        a = db(filt(a, "lowpass", 6000), -3)
        st = wet(a, "memory", 0.3)
        return st if row["type"] != "台词" else st * pan(np.ones(1), p)[0]
    if row["type"] != "台词":
        return pan(db(a, -1.5), 0.0)
    if effect == "无":
        st = wet(a, "room", 0.12)
        return st * pan(np.ones(1), p)[0]
    return pan(a, p)


def gap(prev: dict, row: dict) -> float:
    """两句之间插多少秒静音。剧本给了 pause（这句前面隐去了别人的话，见 scriptize.pauses）就至少留这么久；
    没有这一项的剧本照旧。"""
    if row["type"] != "台词" or prev["type"] != "台词":
        g = GAP["narration"]
    else:
        g = GAP["same"] if row["speaker"] == prev["speaker"] else GAP["change"]
    return max(g, row.get("pause", 0))


def assemble(rows: list[dict], wav_dir: Path, cast: dict, inner: bool,
             marks: list[float] | None = None) -> np.ndarray:
    """marks 不为 None 时，按顺序记下每句的起始秒数（给对照剧本用）。"""
    chunks, prev, pos = [], None, 0
    for row in rows:
        a, sr = sf.read(wav_dir / f"{row['idx']:04d}.wav", dtype="float32")
        assert sr == SR
        p = cast["characters"].get(row["speaker"], {}).get("pan", 0.0)
        st = process(a, row, p, inner)
        if prev is not None:
            n = int(gap(prev, row) * SR)
            chunks.append(np.zeros((n, 2), dtype=np.float32))
            pos += n
        if marks is not None:
            marks.append(pos / SR)
        chunks.append(st.astype(np.float32))
        pos += len(st)
        prev = row
    out = np.concatenate(chunks)
    peak = float(np.abs(out).max())
    if peak > 0.97:
        out *= 0.97 / peak
    return out


def to_mp3(audio: np.ndarray, path: Path, tags: dict, cover: Path | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".wav")
    sf.write(tmp, audio, SR, subtype="PCM_16")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp)]
    if cover and cover.exists():
        cmd += ["-i", str(cover), "-map", "0:a", "-map", "1:v", "-c:v", "mjpeg",
                "-disposition:v", "attached_pic"]
    cmd += ["-ar", "44100", "-c:a", "libmp3lame", "-b:a", "160k", "-joint_stereo", "1",
            "-id3v2_version", "3"]
    for k, v in tags.items():
        cmd += ["-metadata", f"{k}={v}"]
    subprocess.run(cmd + [str(path)], check=True)
    tmp.unlink()
