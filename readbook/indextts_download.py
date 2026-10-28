"""分块并发下载 IndexTTS 2.5 / 2.0 权重和辅助模型（10 号票），不依赖 readbook 包，用系统 Python 跑：

    python -I readbook/indextts_download.py

- 2.5（ModelScope `IndexTeam/IndexTTS-2.5`）→ `<index-tts>/checkpoints/`，2.0（`IndexTeam/IndexTTS-2`）→ `checkpoints_2/`；
  两版的 `qwen0.6bemo4-merge/model.safetensors` 哈希相同，只下一份，硬链接过去；
- 辅助模型从 hf-mirror 只下用得到的文件，放 `checkpoints/hf_cache/`，2.0 通过 `aux_paths` 共用：w2v-bert-2.0
  只要 model.safetensors（不要 conformer_shaw.pt）、BigVGAN 只要 generator、MaskGCT 只要 semantic_codec；
  CAM++ 不在这里下，从 HF 缓存复制 `funasr/campplus` 的 `campplus_cn_common.bin`；
- 单连接只有 ~0.85 MB/s，4–5 路并发能到 ~1.9 MB/s。每个文件切成 64MB 的块，用 curl 的 range 请求下
  （curl 走 schannel，认 Norton 的证书）。已下完的块会跳过，可以反复运行续传。最后校验 sha256。
"""
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.home() / "codes/github.com/index-tts/index-tts"
CK25, CK20 = ROOT / "checkpoints", ROOT / "checkpoints_2"
MS = "https://modelscope.cn/models/{repo}/resolve/master/{path}"
HF = "https://hf-mirror.com/{repo}/resolve/main/{path}"
CHUNK = 64 << 20
WORKERS = int(os.environ.get("DL_WORKERS", 5))


def ms_files(repo):
    out = subprocess.run(["curl", "-sL", f"https://modelscope.cn/api/v1/models/{repo}/repo/files?Recursive=true"],
                         capture_output=True, check=True).stdout
    return [(f["Path"], f["Size"], f["Sha256"]) for f in json.loads(out)["Data"]["Files"] if f["Type"] == "blob"]


def hf_files(repo):
    out = subprocess.run(["curl", "-sL", f"https://hf-mirror.com/api/models/{repo}/tree/main?recursive=true"],
                         capture_output=True, check=True).stdout
    return {f["path"]: (f["size"], (f.get("lfs") or {}).get("oid")) for f in json.loads(out) if f["type"] == "file"}


def plan():
    jobs = []  # (url, dest, size, sha256 or None)
    for path, size, sha in ms_files("IndexTeam/IndexTTS-2.5"):
        jobs.append((MS.format(repo="IndexTeam/IndexTTS-2.5", path=path), CK25 / path, size, sha))
    aux = CK25 / "hf_cache"
    w2v = hf_files("facebook/w2v-bert-2.0")
    for p in ("config.json", "preprocessor_config.json", "model.safetensors"):
        jobs.append((HF.format(repo="facebook/w2v-bert-2.0", path=p), aux / "w2v-bert-2.0" / p, *w2v[p]))
    bv = hf_files("nvidia/bigvgan_v2_22khz_80band_256x")
    for p in ("config.json", "bigvgan_generator.pt"):
        jobs.append((HF.format(repo="nvidia/bigvgan_v2_22khz_80band_256x", path=p), aux / "bigvgan" / p, *bv[p]))
    mg = hf_files("amphion/MaskGCT")
    p = "semantic_codec/model.safetensors"
    jobs.append((HF.format(repo="amphion/MaskGCT", path=p), aux / "semantic_codec_model.safetensors", *mg[p]))
    for path, size, sha in ms_files("IndexTeam/IndexTTS-2"):
        if path == "qwen0.6bemo4-merge/model.safetensors":
            continue  # 和 2.5 那份哈希相同，下完后复制
        jobs.append((MS.format(repo="IndexTeam/IndexTTS-2", path=path), CK20 / path, size, sha))
    return jobs


def fetch(url, part, a, b):
    want = b - a + 1
    if part.exists() and part.stat().st_size == want:
        return want
    for attempt in range(30):
        part.unlink(missing_ok=True)
        r = subprocess.run(["curl", "-sL", "--fail", "--connect-timeout", "30", "--speed-time", "60",
                            "--speed-limit", "20000", "-r", f"{a}-{b}", "-o", str(part), url])
        if r.returncode == 0 and part.exists() and part.stat().st_size == want:
            return want
        time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError(f"failed {url} {a}-{b}")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while blk := f.read(8 << 20):
            h.update(blk)
    return h.hexdigest()


def main():
    jobs = plan()
    total = sum(s for _, _, s, _ in jobs)
    print(f"{len(jobs)} files, {total / 1e9:.2f} GB", flush=True)
    tasks, todo = [], []
    for url, dest, size, sha in jobs:
        if dest.exists() and dest.stat().st_size == size:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        parts = []
        for i, a in enumerate(range(0, max(size, 1), CHUNK)):
            b = min(a + CHUNK, size) - 1
            part = dest.with_name(dest.name + f".part{i:04d}")
            parts.append(part)
            if size:
                tasks.append((url, part, a, b))
        todo.append((dest, size, parts))
    done_bytes = sum(s for _, d, s, _ in jobs if d.exists() and d.stat().st_size == s)
    t0, got = time.time(), 0
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(fetch, *t): t for t in tasks}
        pending_files = list(todo)
        for fu in cf.as_completed(futs):
            got += fu.result()
            el = time.time() - t0
            # 已经下齐的文件立刻拼起来，好让 2.5 尽早能用
            for item in list(pending_files):
                dest, size, parts = item
                if all(p.exists() for p in parts) and sum(p.stat().st_size for p in parts) == size:
                    with open(dest, "wb") as out:
                        for p in parts:
                            out.write(p.read_bytes())
                    for p in parts:
                        p.unlink()
                    pending_files.remove(item)
                    print(f"[done] {dest}", flush=True)
            print(f"{(done_bytes + got) / 1e9:6.2f}/{total / 1e9:.2f} GB  {got / el / 1e6:.2f} MB/s", flush=True)
        for dest, size, parts in pending_files:  # 空文件
            dest.touch()
    bad = []
    for url, dest, size, sha in jobs:
        if sha and dest.stat().st_size > 1 << 20 and sha256(dest) != sha:
            bad.append(dest)
    print("sha256 mismatch:", bad if bad else "none", flush=True)
    src = CK25 / "qwen0.6bemo4-merge/model.safetensors"
    dst = CK20 / "qwen0.6bemo4-merge/model.safetensors"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dst)
    print("ALL DONE", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
