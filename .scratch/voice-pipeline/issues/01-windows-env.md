# 01: 在 5090 上搭好可复现的基础环境

Type: task（HITL）
Status: resolved
Blocked by: 无（可以立即开始）

## Question

在 Windows 的 RTX 5090 笔记本上，把后续所有原型要用的基础环境搭好并验证：
- CUDA 12.8、PyTorch cu128、ffmpeg、uv、Claude Code；
- readbook 仓库能跑；
- 能读到游戏语音包 `VOICE.PAK` 和剧本（中日文），也就是 gameresource 仓库里的 `lscr`、`lacpak`、`daily` 能用；
- 代码里写死的 Mac 路径（游戏目录、gameresource 仓库、IndexTTS2 环境、模型缓存）改成可配置。

**需要人操作的部分**（列成清单交给用户）：系统层面的安装；游戏文件放到哪里；是否开 SSH。

**完成标准**：在 Windows 上，`readbook.wa2.load_day` 能取出〈2002〉的 65 句，并能导出其中任意一句的原声 wav。

## Answer

完成标准已经达到：在 Windows 上，`uv run python -m readbook.wa2 "01 第二部-共通" 2010-11-25` 取出 156 行，其中 `sid ≥ 2002` 的 65 句就是〈2002〉，语音文件名和中文都和仓库里的剧本逐句一致；`uv run python -m readbook.voices 2002_0000_02.OGG 2002_0024_00.OGG` 能导出 48k 单声道的原声 wav。完整的搭建过程和坑写在 `docs/audiobook/小说转有声书运行环境.md`。要点：

- **装了什么**：
  - uv 0.12.19，Python 3.12.14；
  - torch、torchaudio 2.11.0+cu128。GPU 实测正常：sm_120，bf16 大约 54 TFLOPS；
  - ffmpeg 9.0.2，用 winget 装在用户目录；
  - Claude Code 2.1.311，用官方脚本装到 `~/.local/bin`，用户已经登录，`claude -p --json-schema` 测过能用。
  - 不装 CUDA Toolkit，也不装 flash-attn。
- **依赖按平台分开**：`pyproject.toml` 里，Mac 装 mlx-audio；Windows 装 cu128 版 torch（单独的 PyTorch 源）和 truststore（当初装的是 pip-system-certs，05 号票换掉了，见下一条）。另外补上了 scipy（`mix.py` 直接用到它，以前是 mlx-audio 顺带装进来的）。Mac 那边锁定的版本一个都没变。
- **路径配置**（用户定的）：每个平台一份配置，都进 git：`config/windows.yaml`、`config/mac.yaml`，由 `readbook/config.py` 按 `sys.platform` 选。原来写死的 `GAME`、`TOOLS`、`INDEX_DIR`、`COVER` 都改成从这里读。游戏就用现有的安装位置 `C:\Others\WHITE ALBUM2_Chs`，不挪。
- **Norton 换证书**（用户定的）：Norton 不动，每个环境各自改用 Windows 证书库。uv 在 `%APPDATA%\uv\uv.toml` 里设 `system-certs = true`；Python 在 `readbook/config.py` 里调用 `truststore.inject_into_ssl()`（只在 Windows 上）；winget 装软件时要加 `--source winget`。当初 Python 用的是 pip-system-certs，05 号票发现它注入的是 pip 自带的那份 truststore，和 huggingface_hub 2.x 自己用的那份叠在一起，一联网就 `RecursionError`，所以换掉了。以后建 IndexTTS2、Applio 的环境：huggingface_hub 是老版（用 requests）的，可以照旧加 pip-system-certs；是 2.x 的，就在入口调用 `truststore.inject_into_ssl()`。
- **为 Windows 改的代码**：
  - `claude -p` 和 CAM++ 子进程交换文本时显式用 UTF-8（Windows 默认按 GBK）；
  - `wa2` 输出 jsonl 时固定用 UTF-8；
  - CAM++ 用的临时 wav 要先关掉，别的进程才读得到；
  - `voices.py` 加了一条导出原声的命令。
- **SSH**：用户定的是先不开。
- **留给后面的票**：
  - IndexTTS2 仓库和它的环境还没装，CAM++ 要靠它（05 号票会用到，届时也可以考虑把 CAM++ 直接搬进 readbook 的环境，因为现在这边已经有 torch 了）；
  - 用 MLX 合成的代码路径要换成 PyTorch 版，这是 06 号票的事。
- **网速**：大约 0.8 MB/s，镜像和直连差不多。下大模型要放到后台，提前开始。
