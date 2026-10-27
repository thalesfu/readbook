"""本机路径：按平台读 config/ 下的一份（Windows 读 windows.yaml，Mac 读 mac.yaml），两份都进 git。"""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "config" / {"win32": "windows.yaml", "darwin": "mac.yaml"}[sys.platform]
_paths = {k: Path(v).expanduser() for k, v in yaml.safe_load(FILE.read_text(encoding="utf-8")).items()}

GAME = _paths["game"]           # 白色相簿2 安装目录：VOICE.PAK、IC/VOICE.PAK、script.pak、ck-gal.pak
TOOLS = _paths["tools"]         # gameresource 仓库里的 lfpak / lscr / daily / lacpak
INDEX_DIR = _paths["indextts"]  # IndexTTS2 仓库（自带 uv 环境，CAM++ 权重在 checkpoints_2/hf_cache/）
COVER = _paths["cover"]         # MP3 封面
