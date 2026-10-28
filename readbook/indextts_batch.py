"""在 IndexTTS 自己的环境里跑（uv run --project <index-tts>），不依赖 readbook 包。

输入是一份 jobs.json，两种写法：
- bakeoff.cmd_index 写的列表 [{out, text, spk, emo_text}, ...]：2.0，文字描述定情绪；
- index_passage 写的 {"version": "2" | "2.5", "jobs": [{out, text, spk, seed, max_secs,
  emo_vector?, emo_audio?, emo_alpha}, ...]}：情感向量先过官方的 normalize_emo_vec
  （偏置 + 总和压到 0.8 以下，Python API 不会自己做），use_random=False。
  2.5 用 bf16，2.0 用 fp32。辅助模型（w2v-bert、BigVGAN、CAM++ 等）两版共用 checkpoints/hf_cache。
已存在的输出跳过。每条的耗时和时长追加到 jobs.json 旁的 timing.jsonl。
"""

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

TOKENS_PER_SEC = {"2": 50, "2.5": 25}  # 语义 token 速率，生成上限按它换算（试跑实测：code 长度 ÷ 时长）


def load(version: str, emo_text: bool):
    if version == "2.5":
        from indextts.infer_v2_5 import IndexTTS2

        return IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_bf16=True,
                         use_cuda_kernel=False, use_qwen_emo=emo_text)
    from indextts.infer_v2 import IndexTTS2
    from indextts.utils.model_download import ensure_models_available

    return IndexTTS2(cfg_path="checkpoints_2/config.yaml", model_dir="checkpoints_2", use_fp16=False,
                     use_cuda_kernel=False, use_qwen_emo=emo_text,
                     aux_paths=ensure_models_available("checkpoints"))


def main(path: Path):
    spec = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(spec, list):  # bakeoff.cmd_index
        tts = load("2", True)
        for i, j in enumerate(spec):
            if Path(j["out"]).exists():
                continue
            tts.infer(spk_audio_prompt=j["spk"], text=j["text"], output_path=j["out"],
                      emo_text=j["emo_text"], use_emo_text=True, emo_alpha=0.6, verbose=False)
            print(f"index {i + 1}/{len(spec)}", flush=True)
        return

    version, jobs = spec["version"], spec["jobs"]
    todo = [j for j in jobs if not Path(j["out"]).exists()]
    if not todo:
        return
    t = time.perf_counter()
    tts = load(version, False)
    timing = path.with_name("timing.jsonl")
    with timing.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"load_secs": round(time.perf_counter() - t, 2), "version": version}) + "\n")
        for i, j in enumerate(todo):
            kw = {"lang": "ZH"} if version == "2.5" else {}
            if j.get("emo_vector"):
                kw["emo_vector"] = tts.normalize_emo_vec(j["emo_vector"], apply_bias=True)
            elif j.get("emo_audio"):
                kw["emo_audio_prompt"] = j["emo_audio"]
            torch.manual_seed(j["seed"])
            torch.cuda.synchronize()
            t = time.perf_counter()
            tts.infer(spk_audio_prompt=j["spk"], text=j["text"], output_path=j["out"], emo_alpha=j["emo_alpha"],
                      use_random=False, verbose=False, max_mel_tokens=math.ceil(j["max_secs"] * TOKENS_PER_SEC[version]),
                      **kw)
            torch.cuda.synchronize()
            secs = time.perf_counter() - t
            dur = sf.info(j["out"]).duration
            log.write(json.dumps({"out": j["out"], "secs": round(secs, 3), "dur": round(dur, 3),
                                  "chars": j["chars"]}, ensure_ascii=False) + "\n")
            log.flush()
            print(f"index{version} {i + 1}/{len(todo)} {secs:.1f}s/{dur:.1f}s", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
