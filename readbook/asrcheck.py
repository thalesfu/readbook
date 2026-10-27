"""用 Whisper 把合成结果识别回文字，按字错率（CER）找出读错、漏读、念成别的语言的句子。

CER 高不一定是读错：哭腔、拟声词、"………" 这种本来就难识别。所以这里只给排序，最后要人听。

Windows 上用官方 PyTorch 版（TorchWhisper），在本进程里跑。Mac 上用 mlx-audio 版：它偶尔会在
某一句上死循环（和音频内容无关），所以识别放在子进程里，每句限时，超时就杀掉重启，这一句记为
识别超时。结果逐句追加到缓存，中断后重跑不重复。
"""

import json
import multiprocessing as mp
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import hf_model
from .synth import tts_text

STT = "mlx-community/whisper-large-v3-turbo"
PROCESSOR = "openai/whisper-large-v3-turbo"  # mlx-community 的仓库里没带分词器；PyTorch 版直接用这个仓库
PROMPT = "以下是普通话的句子。"
PUNCT = re.compile(r"[\W_]+", re.U)
TIMEOUT = 30


def cer(ref: str, hyp: str) -> float:
    ref, hyp = PUNCT.sub("", ref), PUNCT.sub("", hyp)
    if not ref:
        return 0.0
    prev = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        cur = [i]
        for j, b in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1] / len(ref)


def _worker(inbox: mp.Queue, outbox: mp.Queue):
    from mlx_audio.stt.utils import load_model
    from transformers import WhisperProcessor

    model = load_model(STT)
    model._processor = WhisperProcessor.from_pretrained(PROCESSOR)
    outbox.put("ready")
    while True:
        path = inbox.get()
        # 只用温度 0、不接上文：哭声、喘气会让默认的多温度回退反复重试
        r = model.generate(path, language="zh", initial_prompt=PROMPT,
                           temperature=0.0, condition_on_previous_text=False)
        outbox.put(r.text.strip())


class TorchWhisper:
    """官方 PyTorch 版，fp16 跑在 GPU 上。和 mlx 版一样只用温度 0（贪心）、不接上文。"""

    def __init__(self):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        path = hf_model(PROCESSOR, ["*.json", "*.txt", "*.safetensors"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = WhisperProcessor.from_pretrained(path)
        self.model = WhisperForConditionalGeneration.from_pretrained(path, dtype=self.dtype).to(self.device).eval()
        self.prompt = self.processor.get_prompt_ids(PROMPT, return_tensors="pt").to(self.device)

    def transcribe_audio(self, a: np.ndarray, sr: int) -> str:
        import torch
        import torchaudio

        if a.ndim > 1:
            a = a.mean(axis=1)
        w = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)), sr, 16000)
        feats = self.processor(w.numpy(), sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            ids = self.model.generate(feats.to(self.device, self.dtype), language="zh", task="transcribe",
                                      prompt_ids=self.prompt, max_new_tokens=200)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return text.removeprefix(PROMPT).strip()


class Checker:
    def __init__(self):
        if sys.platform == "win32":
            self.whisper = TorchWhisper()
            return
        self.ctx = mp.get_context("spawn")
        self.proc = None
        self._start()

    def _start(self):
        if self.proc:
            self.proc.kill()
            self.proc.join()
        self.inbox, self.outbox = self.ctx.Queue(), self.ctx.Queue()
        self.proc = self.ctx.Process(target=_worker, args=(self.inbox, self.outbox), daemon=True)
        self.proc.start()
        assert self.outbox.get(timeout=300) == "ready"

    def transcribe(self, path: Path) -> str | None:
        if sys.platform == "win32":
            a, sr = sf.read(path, dtype="float32")
            return self.whisper.transcribe_audio(a, sr)
        self.inbox.put(str(path))
        try:
            return self.outbox.get(timeout=TIMEOUT)
        except Exception:
            self._start()
            return None

    def check(self, rows: list[dict], wav_dir: Path, lexicon: dict, cache: Path) -> list[dict]:
        done = {}
        if cache.exists():
            done = {r["idx"]: r for r in map(json.loads, cache.read_text(encoding="utf-8").splitlines())}
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("a", encoding="utf-8") as fh:
            for row in rows:
                ref = tts_text(row, lexicon)
                if row["idx"] in done or len(PUNCT.sub("", ref)) < 2:  # "………"、"呃…" 这类不查
                    continue
                hyp = self.transcribe(wav_dir / f"{row['idx']:04d}.wav")
                r = {"idx": row["idx"], "speaker": row["speaker"], "ref": ref,
                     "hyp": hyp if hyp is not None else "（识别超时）",
                     "cer": round(cer(ref, hyp), 3) if hyp is not None else None}
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                done[row["idx"]] = r
        return [done[k] for k in sorted(done)]
