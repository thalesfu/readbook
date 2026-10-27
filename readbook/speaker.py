"""音色一致性：用 CAM++ 说话人向量算两段声音像不像同一个人。

实测（白色相簿2 小样）：同一个人 0.7–0.8，不同角色 0.2–0.5。Qwen3-TTS 自带的说话人编码器
区分度太低（男女声都有 0.95），不能用来质检。

Windows 上直接在本进程里用 GPU 算，权重和 IndexTTS2 用的是同一份（HuggingFace 的
funasr/campplus，campplus_cn_common.bin）。Mac 上 readbook 的环境没有 PyTorch，仍然交给
IndexTTS2 环境里的 campplus_worker.py。两边的特征一样：重采样到 16k、80 维 fbank、减均值。
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import INDEX_DIR, hf_model

WORKER = Path(__file__).with_name("campplus_worker.py")


class SpeakerSim:
    def __init__(self):
        self.cache: dict[str, np.ndarray] = {}
        self.proc = self.model = None
        if sys.platform == "win32":
            import torch
            from funasr.models.campplus.model import CAMPPlus

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = CAMPPlus(feat_dim=80, embedding_size=192)
            weights = hf_model("funasr/campplus", ["campplus_cn_common.bin"]) / "campplus_cn_common.bin"
            self.model.load_state_dict(torch.load(weights, map_location="cpu"))
            self.model.eval().to(self.device)
            return
        self.proc = subprocess.Popen(
            ["uv", "run", "--project", str(INDEX_DIR), "python", str(WORKER)],
            cwd=INDEX_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1)
        assert self.proc.stdout.readline().strip() == "ready"

    def embed(self, path: str | Path) -> np.ndarray:
        key = str(path)
        if key not in self.cache:
            if self.proc:
                self.proc.stdin.write(key + "\n")
                self.cache[key] = np.array(json.loads(self.proc.stdout.readline()))
            else:
                a, sr = sf.read(key, dtype="float32")
                self.cache[key] = self.embed_audio(a, sr)
        return self.cache[key]

    def embed_audio(self, a: np.ndarray, sr: int) -> np.ndarray:
        if self.model is None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete_on_close=False) as f:
                f.close()
                sf.write(f.name, a, sr)
                self.proc.stdin.write(f.name + "\n")
                return np.array(json.loads(self.proc.stdout.readline()))
        import torch
        import torchaudio

        if a.ndim > 1:
            a = a.mean(axis=1)
        w = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None],
                                           sr, 16000)
        f = torchaudio.compliance.kaldi.fbank(w, num_mel_bins=80, dither=0, sample_frequency=16000)
        f = f - f.mean(dim=0, keepdim=True)
        with torch.no_grad():
            e = self.model(f[None].to(self.device)).squeeze().cpu().numpy()
        return e / np.linalg.norm(e)

    def sim(self, a, b) -> float:
        return float(a @ b)
