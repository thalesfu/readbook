"""CAM++ 说话人向量服务，在 IndexTTS2 的环境里跑（那边有 PyTorch 和 campplus_cn_common.bin）。

协议：stdin 每行一个 wav 路径，stdout 每行回一个 JSON 数组（192 维、已归一化的向量）。
不依赖 readbook 包。
"""

import json
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus

model = CAMPPlus(feat_dim=80, embedding_size=192)
model.load_state_dict(torch.load("checkpoints_2/hf_cache/campplus_cn_common.bin", map_location="cpu"))
model.eval()
print("ready", flush=True)
for line in sys.stdin:
    a, sr = sf.read(line.strip(), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    w = torchaudio.functional.resample(torch.from_numpy(a)[None], sr, 16000)
    f = torchaudio.compliance.kaldi.fbank(w, num_mel_bins=80, dither=0, sample_frequency=16000)
    f = f - f.mean(dim=0, keepdim=True)
    with torch.no_grad():
        e = model(f[None]).squeeze().numpy()
    print(json.dumps((e / np.linalg.norm(e)).tolist()), flush=True)
