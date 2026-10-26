"""音色一致性：用 CAM++ 说话人向量算两段声音像不像同一个人。

实测（白色相簿2 小样）：同一个人 0.7–0.8，不同角色 0.2–0.5。Qwen3-TTS 自带的说话人编码器
区分度太低（男女声都有 0.95），不能用来质检。
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

INDEX_DIR = Path("~/codes/github.com/index-tts/index-tts").expanduser()
WORKER = Path(__file__).with_name("campplus_worker.py")


class SpeakerSim:
    def __init__(self):
        self.proc = subprocess.Popen(
            ["uv", "run", "--project", str(INDEX_DIR), "python", str(WORKER)],
            cwd=INDEX_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        assert self.proc.stdout.readline().strip() == "ready"
        self.cache: dict[str, np.ndarray] = {}

    def embed(self, path: str | Path) -> np.ndarray:
        key = str(path)
        if key not in self.cache:
            self.proc.stdin.write(key + "\n")
            self.cache[key] = np.array(json.loads(self.proc.stdout.readline()))
        return self.cache[key]

    def embed_audio(self, a: np.ndarray, sr: int) -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, a, sr)
            self.proc.stdin.write(f.name + "\n")
            return np.array(json.loads(self.proc.stdout.readline()))

    def sim(self, a, b) -> float:
        return float(a @ b)
