# 交接：在 Windows 5090 上继续 voice-pipeline 地图

写于 2026-10-26，交接自 Mac 上的会话。目标会话运行在 Windows 的 RTX 5090 笔记本（24GB，CUDA 12.8）上。

## 这个会话要做什么

接着走 `.scratch/voice-pipeline/map.md` 这张 wayfinder 地图，依次做：

1. **在 5090 上搭好可复现的基础环境**（`issues/01-windows-env.md`，要用户配合）；
2. **评测工具补强**（`issues/05-eval-tools.md`）；
3. **在 5090 上复现混合版**（`issues/06-reproduce-hybrid.md`）。这是后面所有对比的基线。

一个会话只解决一张票。每张票开工前先认领（把 `Status` 改成 `claimed` 并保存），解决后按 `docs/agents/issue-tracker.md` 的"Wayfinding operations"回写答案、改成 `resolved`，并在地图的 Decisions so far 加一行。

## 先读这些（不在这里重复）

- **地图和票**：`.scratch/voice-pipeline/map.md`，以及 `.scratch/voice-pipeline/issues/`。
- **术语**：`GLOSSARY.md`，统一用定妆音、音色锚点、情绪定妆音、讲述模式、音色门槛、小样、混合版。
- **背景、决策和数据**：`docs/audiobook/` 下三篇：
  - `小说转有声书技术报告.md`：现状、问题、业界调研、候选尝试、评估方法；
  - `小说转有声书方案.md`：已定事项和决策理由；
  - `小说转有声书试做结果.md`：第一轮试做的数据。
- **三份调研笔记**，在三个分支上，用 `git show <分支>:<文件>` 直接读：
  - `research/qwen3-torch:research/qwen3-torch.md`
  - `research/rvc:research/rvc.md`
  - `research/indextts25:research/indextts25.md`
- **混合版的定妆音**：`books/wa2/voices/`，索引见 `voices.yaml`。
- **基线试听音频和对照剧本**：`books/wa2/baseline/`。其中"小样-对照剧本.md"的时间戳是按锚点版算的，和混合版前后差几秒。

## 换机器要注意的事

**代码是在 Mac 上写的，有两类依赖在 Windows 上不能直接用**：
- **MLX**：所有 `from mlx_audio...` 和 `mlx-community/...` 模型 ID 都只能在 Apple Silicon 上跑。涉及 `readbook/synth.py`、`bakeoff.py`、`passage.py`、`asrcheck.py`。Windows 上要换成官方 PyTorch 版，Qwen3-TTS 的做法见 `research/qwen3-torch` 的调研笔记。
- **写死的 Mac 路径**，要改成可配置（这是"搭好基础环境"那张票的一部分）：
  - `readbook/wa2.py`：`GAME`（游戏安装目录，Mac 上是 `~/Windows/games/WHITE ALBUM2_Chs`）和 `TOOLS`（gameresource 仓库里的工具目录）；
  - `readbook/bakeoff.py`、`readbook/speaker.py`：`INDEX_DIR`（IndexTTS2 仓库，CAM++ 的权重是它的 `checkpoints_2/hf_cache/campplus_cn_common.bin`）；
  - `readbook/trial.py`：`COVER`（Calibre 书库里的封面）。

**需要准备的数据和仓库**：
- **游戏**：用到 `VOICE.PAK`、`IC/VOICE.PAK`、`script.pak`（日文）、`ck-gal.pak`（中文）。
- **gameresource 仓库**：`git@github.com:thalesfu/gameresource.git`。用到的是 `acg/白色相簿2/Dev/tools/` 下的 `lfpak`、`lscr`、`daily`、`lacpak`、`cnenc`，以及 `Dev/data/汉化码表.txt`。
- **IndexTTS2 仓库和模型**：github.com/index-tts/index-tts。模型从 ModelScope 下（`IndexTeam/IndexTTS-2`、`IndexTeam/IndexTTS-2.5`）。

**网络（国内）**：
- PyPI 走清华镜像，`pyproject.toml` 里已经配好；直连只有约 16 KB/s。
- HuggingFace 走 `HF_ENDPOINT=https://hf-mirror.com`。遇到 xet 报 401 时，再加 `HF_HUB_DISABLE_XET=1`。
- IndexTTS 系列在 hf-mirror 上没有镜像，会跳回 huggingface.co，要用 ModelScope。

**已知的坑**：
- **Qwen3 官方版**：必须用非流式模式（`non_streaming_mode=True`），不然越读越快；子 talker 的温度和 repetition_penalty 要另外对齐 mlx 版。详见调研笔记。
- **Qwen3 复刻模式不能调语速**：锚点语速一偏，就得事后大幅变速，声音会发僵。这是挑锚点时要按音高、语速来挑的原因。
- **剧本化**：`claude -p --output-format json --json-schema ...`，读 `structured_output`。Windows 上也要装好 Claude Code 才能用。
- **合成失控**：带指令或提示的生成，遇到"………"这类句子会一直生成到上限。现在的保护是：只有标点的句子插静音；超时长的句子判为失控并重做。批量生成时要按句长设生成上限。

## 用户的偏好

- **语言**：始终用中文交流。
- **拷问**：一次只问一个问题，并给出推荐答案；决定由用户做，能查到的事实自己查。
- **交付**：每出一版都附上 MP3、带时间戳的对照剧本，以及指标对比（音高和语速范围、声纹相似度、认人准确率）。用户靠试听判断，指标只当门槛。
- **文档**：都在仓库的 `docs/audiobook/`。Obsidian 原处只留指向仓库的短页，不要再往 Obsidian 写这个项目的笔记。
- **提交和推送**：等用户点头再做。
- **skill 同步**：新建或修改用户自己写的 skill 后，按用户全局 CLAUDE.md 的规则同步到 agentskills 仓库。Matt Pocock 这套 skill 不在同步范围内。

## 建议调用的 skill

- `wayfinder`：主线，走地图、选票、认领、回写。
- `research`：处理调研票，或者某张票缺一个事实时。
- `prototype`：原型票，也就是合成小样、出对比。
- `grilling` 和 `domain-modeling`：拷问票（比如最后"写定正式合成流程规格"那张），以及出现新术语的时候。
- `diagnosing-bugs`：合成、安装、复现出问题时。
- `setup-matt-pocock-skills`：Windows 上还没装这套 skill，或者要重新配置时。
- `writing-for-agents`：要改 CLAUDE.md 或 AGENTS.md 时。
