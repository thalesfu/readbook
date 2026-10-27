"""本机路径：按平台读 config/ 下的一份（Windows 读 windows.yaml，Mac 读 mac.yaml），两份都进 git。

另外管联网：
- Windows 上 Norton 会把 HTTPS 证书换成它自己签的，这里让 Python 的 ssl 改认 Windows 证书库；
- HuggingFace 国内直连太慢，走 hf-mirror；xet 在镜像上会报 401，关掉。
"""

import os
import sys
from pathlib import Path

import yaml

if sys.platform == "win32":
    import truststore

    truststore.inject_into_ssl()
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 要在 huggingface_hub 导入之前设
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "config" / {"win32": "windows.yaml", "darwin": "mac.yaml"}[sys.platform]
_paths = {k: Path(v).expanduser() for k, v in yaml.safe_load(FILE.read_text(encoding="utf-8")).items()}

GAME = _paths["game"]           # 白色相簿2 安装目录：VOICE.PAK、IC/VOICE.PAK、script.pak、ck-gal.pak
TOOLS = _paths["tools"]         # gameresource 仓库里的 lfpak / lscr / daily / lacpak
INDEX_DIR = _paths["indextts"]  # IndexTTS2 仓库（Mac 上 CAM++ 跑在它的环境里）
COVER = _paths["cover"]         # MP3 封面


def hf_model(repo: str, files: list[str] | None = None) -> Path:
    """HuggingFace 模型的本地目录：缓存里有就直接用，不联网；没有再下载（大模型最好提前在后台下好）。"""
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo, allow_patterns=files, local_files_only=True))
    except Exception:
        return Path(snapshot_download(repo, allow_patterns=files))
