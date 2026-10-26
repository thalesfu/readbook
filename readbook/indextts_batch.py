"""在 IndexTTS2 自己的环境里跑（uv run --project <index-tts>），不依赖 readbook 包。

输入是 bakeoff.cmd_index 写的 jobs.json：[{out, text, spk, emo_text}, ...]，已存在的输出跳过。
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from indextts.infer_v2 import IndexTTS2  # noqa: E402

jobs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
tts = IndexTTS2(cfg_path="checkpoints_2/config.yaml", model_dir="checkpoints_2", use_fp16=False)
for i, j in enumerate(jobs):
    if Path(j["out"]).exists():
        continue
    tts.infer(spk_audio_prompt=j["spk"], text=j["text"], output_path=j["out"],
              emo_text=j["emo_text"], use_emo_text=True, emo_alpha=0.6, verbose=False)
    print(f"index {i + 1}/{len(jobs)}", flush=True)
