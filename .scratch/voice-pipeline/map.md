# 地图：稳住角色音色的正式合成流程

Label: wayfinder:map

## Destination

一份写定的**正式合成流程规格**，放在 `docs/audiobook/`：用哪些模型、怎么串、参数和质检门槛各是多少。它要在小样和〈2002〉上通过试听：
- 同一角色前后不再"像另一个人"；
- 能分清谁在说；
- 旁白和台词分得开；
- 语气大体贴性格。

同时，每句都要过音色门槛（不低于本人真实原声之间相似度的 P5），认人准确率 ≥95%，按实测速度全书一周内能跑完。

## Notes

- **背景先读**：
  - `docs/audiobook/小说转有声书技术报告.md`：现状、问题、业界调研、候选尝试；
  - `docs/audiobook/小说转有声书方案.md`：已定事项和决策理由；
  - `GLOSSARY.md`：统一术语，用"定妆音""音色锚点""情绪定妆音""讲述模式""音色门槛""小样""混合版"。
- **本图允许做实验**：原型票要真的合成小样、跑评测、出 MP3 给人听，产物用来支撑决策。不做全书生产。
- **验证范围**：只在春希、雪菜、依绪、武也四个人身上验证，但流程必须能直接套到其他角色，不许按人手工特调。
- **机器分工**：以 Windows 上的 RTX 5090 笔记本（24GB，CUDA 12.8）为主机，合成、训练、评测都用 CUDA 版。Mac 上的 MLX 代码（`readbook/` 下现有模块）只作参考。剧本化用 `claude -p`，两边都能跑。
- **对比基准**：每个原型都在**小样**（`readbook/bakeoff.py` 的 `sample_rows`：剧本 2024 第 0–13 句、第 54–84 句）和 **〈2002〉**（2010-11-25，剧本 2002 起 65 句）上，和**混合版**比。
- **混合版的定妆音**在 `books/wa2/voices/`，索引见 `voices.yaml`。
- **评估方法**：见技术报告第七节。试听为准；音色门槛只拦明显不像的，不用来挑最像的。
- **硬约束**：只给自己听；不把声优原声上传到任何云服务。
- **每张票都可能用到的 skill**：`research`（调研票）、`prototype`（原型票）、`grilling` + `domain-modeling`（拷问票）、`diagnosing-bugs`（排查合成问题时）。

## Decisions so far

<!-- 每张关掉的票一行：标题链接 + 一句结论 -->

- [在 5090 上复现混合版](issues/06-reproduce-hybrid.md)：复现成功，用户试听"基本等同"，整体 5090 版更好一些，问题句位置和 Mac 差不多但更明显；基线在 `books/wa2/baseline/*-混合版-5090.*`（小样过门槛 20/30、〈2002〉33/43，认人准确率略高于 Mac），用 `readbook.passage 小样|"2010-11-25 〈2002〉"` 重做；官方 PyTorch 版要对齐 mlx：非流式、子 talker 同温度、重复惩罚 1.5@64，并且**语音编码器必须用 fp32**（bf16 下雪菜一半候选跑偏）；参考码的微小差异会大幅改变复刻稳定性，方向因人而异（春希 bf16 编码更稳），可供 07/08 试；实测每字 0.49 秒，全书约 9.6 天，还没到一周。
- [评测工具补强](issues/05-eval-tools.md)：`uv run python -m readbook.evaluate 小样|2002 <wav 目录或 MP3>` 一条命令出全部指标，Mac 混合版的基线在 `books/wa2/baseline/*-指标.md`（小样 29 句过门槛 21、〈2002〉44 句过 35，最弱是春希）；音色门槛取本人原声和原声中心相似度的 P5（春希 0.45、雪菜 0.40、依绪 0.59、武也 0.55）；CAM++、Whisper、emotion2vec+ 都在 readbook 环境里用 GPU 跑，不再依赖 IndexTTS2；pip-system-certs 和新版 huggingface_hub 冲突，readbook 改用 truststore；基线是从混好的 MP3 算的，06 要拿自己的 MP3 同样跑一遍再比。
- [在 5090 上搭好可复现的基础环境](issues/01-windows-env.md)：Windows 上已经能读游戏剧本和原声，〈2002〉65 句对上了；torch 2.11 cu128、ffmpeg、Claude Code 都装好并验证过；本机路径放在 `config/windows.yaml`、`config/mac.yaml`，按平台自动选；Norton 会换证书，每个 Python 环境都要装 pip-system-certs；IndexTTS2 还没装。细节见 `docs/audiobook/小说转有声书运行环境.md`。
- [Qwen3-TTS 官方 PyTorch 版在 Windows 上推理](issues/04-qwen3-torch-research.md)：装 torch cu128 加 `qwen-tts`，用 sdpa、不装 flash-attn；必须用非流式模式防止越读越快；子 talker 温度和 repetition_penalty 要另外对齐 mlx 版；没有语速参数；速度数字都还没实测。
- [RVC 在 50 系 Windows 上怎么训、怎么用](issues/03-rvc-research.md)：首选 Applio（cu128）；每个角色先用约 45 分钟干净原声训第一版、剔除极端情绪句；推理从 rmvpe、`index_rate` 0.3、`protect` 0.33 起步；日语训练转中文机制上可行，但 `index_rate` 要低，用字错率把关；情绪和字错率的影响要自测。
- [IndexTTS-2.5 在 5090 上怎么用](issues/02-indextts25-research.md)：2.5 已公开，多了时长控制、只支持 bf16；情绪向量是混入样本库示例，权重总和要压到 0.8 以下才不漏音色；"害羞""轻声"改用角色自己的参考音；社区反馈 2.5 中文不如 2.0，原型要两版都做。

## Not yet specified

- **按角色微调 TTS**：如果 RVC 不能把音色稳住，或者伪影太重，要不要改做按角色微调（VoxCPM2 LoRA、GPT-SoVITS、Qwen3 SFT），选哪个。要等"RVC 能不能把音色彻底稳住"的结论。
- **商用 TTS 加 RVC**：商用 TTS 负责情绪和稳定，本地 RVC 转成原作音色。要等"RVC 能不能把音色彻底稳住"和"商用 TTS 对照"两张票。
- **依绪和雪菜要不要进一步拉开**：两人声纹最接近（0.53）。要等音色稳住之后重新试听再定。

## Out of scope

- 普通小说的说话人判断（盲测 93%）：这张图只管声音，另开一张图。
- 全书生产：流程定下后，用 `/to-spec`、`/to-tickets` 另行拆执行任务。
- 其他 7 个主要角色的逐一验证：留给全书生产阶段。
