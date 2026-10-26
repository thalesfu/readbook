"""从游戏语音包里取复刻用的参考音频。

- 角色参考：按语音编号挑几句情绪平稳、长度适中的台词，拼成一段 12–18 秒的音频，
  同时拼出对应的日文逐字稿（VoxCPM2 的"带文本复刻"要用）。
- 逐句原声：把某一天台词各自的原声导出来，供"逐句情绪参考"那一版用。
"""

import io
import re
import sys

import numpy as np
import soundfile as sf

from . import wa2
from .wa2 import GAME

sys.path.insert(0, str(wa2.TOOLS))
from daily import ROUTES, walk  # noqa: E402
from lacpak import LacPak  # noqa: E402
from lfpak import Pak  # noqa: E402

SR = 48000
CALM = re.compile(r"^“[^！？!?…ー～♪]{12,60}”$")  # 不带感叹、拖音、省略的整句（render 已把「」换成“”）


class VoicePak:
    def __init__(self):
        self.paks = [LacPak(GAME / "VOICE.PAK"), LacPak(GAME / "IC" / "VOICE.PAK")]
        self.index = {}
        for p in self.paks:
            for e in p.entries:
                self.index.setdefault(e.name, (p, e))

    def audio(self, name: str) -> np.ndarray:
        """读出一条语音，转成 48kHz 单声道 float32。"""
        p, e = self.index[name]
        data, sr = sf.read(io.BytesIO(p.read(e)), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        if sr != SR:
            n = int(round(len(mono) * SR / sr))
            mono = np.interp(np.linspace(0, len(mono), n, endpoint=False), np.arange(len(mono)), mono)
        return mono.astype(np.float32)


def candidates(code: str, routes=("01 第二部-共通", "02 第二部-雪菜线", "07 第三部-共通前半")):
    """在若干路线里找某个编号的平稳台词：(语音文件, 日文)。"""
    jp = Pak(GAME / "script.pak")
    seen = set()
    for folder, _, chain in ROUTES:
        if folder not in routes:
            continue
        for sid in chain:
            for ev in walk(jp, sid, "cp932"):
                if ev[0] != "line" or not ev[3] or not ev[3].endswith(f"_{code}.OGG"):
                    continue
                text = wa2.clean(ev[2]).replace("\n", "")
                if ev[3] not in seen and CALM.match(text):
                    seen.add(ev[3])
                    yield ev[3], text.strip("“”")


def build_reference(vp: VoicePak, code: str, target=(12.0, 18.0), gap=0.35):
    """挑 3–4 句时长 3–7 秒、响度居中的台词拼起来。返回 (音频, 逐字稿, 用到的文件)。"""
    pool = []
    for name, text in candidates(code):
        a = vp.audio(name)
        dur = len(a) / SR
        if 3.0 <= dur <= 7.0:
            rms = float(np.sqrt(np.mean(a ** 2)))
            pool.append((name, text, a, dur, rms))
        if len(pool) >= 60:
            break
    if not pool:
        raise ValueError(f"编号 {code} 找不到合适的参考台词")
    # 响度离中位数越近越平稳
    med = float(np.median([p[4] for p in pool]))
    pool.sort(key=lambda p: abs(np.log(p[4] / med)))
    picked, total = [], 0.0
    for p in pool:
        if total + p[3] > target[1]:
            continue
        picked.append(p)
        total += p[3] + gap
        if total >= target[0]:
            break
    sil = np.zeros(int(gap * SR), dtype=np.float32)
    audio = np.concatenate([x for p in picked for x in (p[2], sil)][:-1])
    peak = float(np.abs(audio).max()) or 1.0
    audio = audio / peak * 0.89
    text = "".join(p[1] if p[1][-1] in "。！？…" else p[1] + "。" for p in picked)
    return audio, text, [p[0] for p in picked]
