# 调研：RVC 在 50 系 Windows 上怎么训、怎么用

- 对应票：`.scratch/voice-pipeline/issues/03-rvc-research.md`
- 调研日期：2026-10-26
- 场景：中文多角色有声书，个人自用。每个角色有几千到上万句干净的日语游戏单人录音（OGG，44.1/48 kHz），中文由 Qwen3-TTS / IndexTTS 合成，问题是同一角色的音色逐句漂移，想用 RVC 统一音色。
- 标注：**【事实】** 指一手来源（源码、官方文档、论文、release 页）里能直接查到的内容，后面附链接；**【推测】** 指根据这些事实推出来、还没有实测验证的判断。

本文读过的源码版本：
- RVC-Project WebUI `main` @ [`81eed5e`](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/tree/81eed5e8f68b6bed1789f682fe78cdd324495afc)（2026-08-04）
- Applio `main` @ [`6f69726`](https://github.com/IAHispano/Applio/tree/6f69726b140c615c8cc6831a7db8b7c5e48ba8af)（2026-10-26）

---

## 1. 选哪个实现

### 1.1 现状

| | RVC-Project WebUI（上游） | Applio（IAHispano） | Seed-VC |
|---|---|---|---|
| 维护状态 | 【事实】2026 年重新活跃：最新 release 是 [2.3.260718](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/releases/tag/2.3.260718)，最后一次 push 在 2026-08-04 | 【事实】最后一次 push 在 2026-10-26，但 README 写明进入低频维护期："Applio will no longer receive frequent updates"，之后以安全补丁、依赖更新为主（[README](https://github.com/IAHispano/Applio)） | 【事实】仓库已 archived（GitHub API `archived=true`，最后一次 push 在 2025-04-20） |
| 50 系支持 | 【事实】release 提供 `RVC20260718Nvidia50x0.7z`（标注 ">=50x0"）整合包；源码安装时 README 的硬件表里，"NVIDIA RTX 50 系"一行对应 `torch==2.7.1+cu128` + `requirments_cu128_py312.txt`（[README](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/81eed5e8f68b6bed1789f682fe78cdd324495afc/README.md)） | 【事实】`run-install.bat` 用 Miniconda 建 Python 3.12 环境，用 `uv pip` 安装依赖，额外索引是 `https://download.pytorch.org/whl/cu128`；`requirements-engine.txt` 固定 `torch==2.11.0` | — |
| 训练采样率 | 【事实】训练页只能选 `40k` / `48k`，默认 40k（[webui.py L2188-2193](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/81eed5e8f68b6bed1789f682fe78cdd324495afc/webui.py#L2188-L2193)） | 【事实】可选 32000/40000/48000，默认 40000；HiFi-GAN 有 32/40/48k 底模，RefineGAN 只有 24/32k 底模（[prerequisites_download.py L183-205](https://github.com/IAHispano/Applio/blob/6f69726b140c615c8cc6831a7db8b7c5e48ba8af/rvc/lib/tools/prerequisites_download.py#L183-L205)） | — |
| 音高算法 | 【事实】推理：pm / rmvpe / fcpe，默认 rmvpe；训练特征提取只支持 pm / rmvpe（[webui.py L955-956、L1875-1878](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/81eed5e8f68b6bed1789f682fe78cdd324495afc/webui.py#L1875-L1878)）。harvest 和 crepe 已经删除 | 【事实】推理：crepe / rmvpe / fcpe / swift 以及几种 hybrid 组合，默认 rmvpe；训练特征提取：crepe / rmvpe / fcpe | — |
| 内容编码器 | 【事实】hubert_base（实际就是 ContentVec，见 §6） | 【事实】contentvec（默认）/ spin / spin-v2 / chinese-hubert-base / japanese-hubert-base / korean-hubert-base / custom（[core.py](https://github.com/IAHispano/Applio/blob/6f69726b140c615c8cc6831a7db8b7c5e48ba8af/core.py)） | — |
| 命令行与批处理 | 【事实】有完整的 CLI 训练和离线推理文档 [docs/cn/cli.md](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/81eed5e8f68b6bed1789f682fe78cdd324495afc/docs/cn/cli.md)；`infer/cli.py` 的 `--input` 可以传目录；支持**多说话人单模型**（ID 0~109） | 【事实】`core.py` 是 click CLI，有 `preprocess / extract / train / index / infer / batch_infer / tts` 等子命令 | — |
| 许可 | MIT | MIT，另有 Terms of Use | — |

### 1.2 结论

- 【推测】两者都能用。我们的需求是"脚本化批量处理几十个角色、几万句"，更看重 CLI 好不好用、能不能复现，而不是 UI 功能多不多。
- **首选 Applio**。CLI 子命令齐全，embedder 可选（可以试 chinese-hubert-base），带 proposed pitch 自动移调，用 cu128 + torch 2.11 新栈。只要选 **"RVC" 架构 + HiFi-GAN 声码器**，训出来的模型就是标准 RVC v2 `.pth` + `.index`。Applio 的 UI 文案写明 HiFi-GAN "compatible with all clients"（[tabs/train/train.py L380-410](https://github.com/IAHispano/Applio/blob/6f69726b140c615c8cc6831a7db8b7c5e48ba8af/tabs/train/train.py#L380-L410)），所以 Applio 哪天停更了，模型也能直接换到上游 RVC 去推理。
- **上游 RVC 2.3 是可靠的备选**。50 系整合包解压即用，最省事；多说话人单模型也许适合角色多的情况。缺点是 Gradio 3.14 + torch<2.8 的老栈，embedder 只有一种。
- 不推荐 RefineGAN / MRF HiFi-GAN。它们只能在 Applio 里用，RefineGAN 也没有 40/48k 底模。
- Seed-VC 已经归档，不作为主线，只在 §5 借用它的评测数据。

---

## 2. Windows + RTX 5090 笔记本（Blackwell，cu128）安装

### 2.1 事实

- 【事实】PyTorch 从 2.7 开始支持 Blackwell，并提供 CUDA 12.8 预编译 wheel。官方博客原话："support for the NVIDIA Blackwell GPU architecture and pre-built wheels for CUDA 12.8"（[PyTorch 2.7 blog](https://pytorch.org/blog/pytorch-2-7/)）。因此 50 系必须用 **cu128（或更新）构建的 torch ≥ 2.7**。cu118 / cu121 的 wheel 不包含 sm_120 内核。
- 【事实】**上游 RVC 源码安装**（Python 3.12 x64，见 [README](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/81eed5e8f68b6bed1789f682fe78cdd324495afc/README.md)）：
  ```powershell
  py -3.12 -m venv .venv; .venv\Scripts\activate
  python -m pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
  python -m pip install -r requirments_cu128_py312.txt
  ```
  `requirments_cu128_py312.txt` 文件头注明 "Torch must stay below 2.8"，并固定 `onnxruntime-gpu>=1.19.2,<1.20`（CUDA 12 / cuDNN 9 ABI）和 `gradio>=3.14,<3.15`。模型（hubert_base、rmvpe.pt、pretrained_v2、mute.zip）要另外从 [lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI) 下载。README 里给了 `hf download` 命令。
- 【事实】**上游 RVC 整合包**：[2.3.260718 release](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/releases/tag/2.3.260718) 的 `RVC20260718Nvidia50x0.7z` 是给 50 系及以后显卡用的完整包，release 说明里没写它内置的 torch/CUDA 版本。这个版本同时修复了 Windows 训练进程的 libuv 兼容问题，单卡训练也不再走 DDP。
- 【事实】**Applio 源码安装**：`run-install.bat` 会建 Python 3.12 的 conda 环境，再用 `uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match` 安装，依赖里是 `torch==2.11.0`。Windows 上还会装 `onnxruntime-gpu==1.22.0` 和 `nvidia-*-cu12` 运行库（[requirements-engine.txt](https://github.com/IAHispano/Applio/blob/6f69726b140c615c8cc6831a7db8b7c5e48ba8af/requirements-engine.txt)）。启动脚本是 `run-applio.bat`。另外还有桌面版（Install / Repair 自动装环境），以及 Hugging Face 上的 Windows 编译 zip（[安装文档](https://docs.applio.org/getting-started/installation/)）。安装文档没有专门写 50 系。
- 【事实】RVC 推理时 HuBERT 和 F0 都在 16 kHz 上计算（[infer/vc/pipeline.py L54](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/81eed5e8f68b6bed1789f682fe78cdd324495afc/infer/vc/pipeline.py#L54)，Applio 也一样）。所以 TTS 输出是 24k 还是 48k 不影响内容特征，输出采样率由模型决定。

### 2.2 推测与注意事项

- 【推测】Applio 依赖的 torch 2.11 + cu128 比 RVC 的 2.7.1 新，对 Blackwell 的支持只会更成熟。笔记本 5090 的驱动要装到支持 CUDA 12.8 以上的版本。
- 【推测】装好后先跑 `python -c "import torch;print(torch.cuda.get_device_capability())"`，确认输出 `(12, 0)`、`torch.cuda.is_available()` 为 True，再开始训练。
- 【推测】安装路径用纯 ASCII、不带空格的目录，避免 conda、ffmpeg、faiss 的路径问题。RVC 2.3 声称修好了编码问题，但没必要冒这个险。

---

## 3. 训练数据

### 3.1 时长

- 【事实】上游 FAQ：推荐 **10~50 分钟**；音质好、音色辨识度高时 5~10 分钟也行；1~2 分钟有人成功但难以复现；不建议少于 1 分钟（[docs/cn/faq.md Q10](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/81eed5e8f68b6bed1789f682fe78cdd324495afc/docs/cn/faq.md)）。上游 README 写的是"推荐至少收集 10 分钟低底噪语音数据"。
- 【事实】FAQ Q9 讲轮数：底噪大、质量低的数据 20~30 轮就够；数据干净、量大时可以加，"200 是 ok 的"。
- 【事实】FAQ Q11 讲 index：训练集质量高、时长长、total_epoch 也高时，index_rate 的作用会变小，甚至可以不用 index。
- 【事实】Applio 文档的说法是 "There is no fixed duration that guarantees quality"：干净、多样的小数据集可以胜过重复的大数据集（[Prepare a dataset](https://docs.applio.org/guides/how-create-datasets/)）。
- 【事实】索引规则：特征向量超过 2×10⁵ 个时，会用 MiniBatchKMeans 压到 1 万个中心（[Applio extract_index.py L67-78](https://github.com/IAHispano/Applio/blob/6f69726b140c615c8cc6831a7db8b7c5e48ba8af/rvc/train/process/extract_index.py#L67-L78)）。内容特征是 50 帧/秒（16 kHz、hop 320），所以大约 1.1 小时以上的数据集，索引都会被聚类压缩。
- 【推测】雪菜有 8600 句，春希有 1.8 万句。按游戏台词平均 2~4 秒估算，**单个角色有 5~20 小时**，远超 FAQ 推荐的上限。数据量不是瓶颈，关键是**挑选**。建议先挑 **30~60 分钟**情绪平稳到中等、覆盖多种句式的台词训第一版；再用 2~3 小时训一版做对照。数据越多，每轮步数越多，同样轮数下训练时间成倍增长，收益是否递减要靠 §5.3 的指标实测，目前没有一手数据。

### 3.2 切片

- 【事实】上游 RVC 预处理（[train/preprocess.py L37-52](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/81eed5e8f68b6bed1789f682fe78cdd324495afc/train/preprocess.py#L37-L52)）分三步：
  1. 先用 openvpi audio-slicer 按静音切分，参数是 `threshold=-42 dB`、`min_length=1500 ms`、`min_interval=400 ms`、`max_sil_kept=500 ms`；
  2. 再按 `per=3.7 s`、重叠 0.3 s 切成定长段；
  3. 每段做峰值归一（0.9×0.75 混合），加 48 Hz 高通滤波。
- 【事实】Applio 的 `preprocess` 有三种模式：`Automatic`（默认，用上面那套 slicer，定长 3.0 s、重叠 0.3 s）、`Simple`（按 `chunk_len` 0.5~5.0 s、默认 3.0 s 直接切）、`Skip`（不切）。归一化模式可选 none / pre / post，UI 默认 post（[core.py](https://github.com/IAHispano/Applio/blob/6f69726b140c615c8cc6831a7db8b7c5e48ba8af/core.py)、[preprocess.py](https://github.com/IAHispano/Applio/blob/6f69726b140c615c8cc6831a7db8b7c5e48ba8af/rvc/train/preprocess/preprocess.py)）。
- 【事实】FAQ Q14：显存不够时，把训练音频手工切短一些。
- 【推测】游戏台词本来就是一句一个文件，不需要手工切。把选好的 OGG 先解码成 WAV（或直接交给预处理，两者都能读），用默认的 Automatic 模式即可。去掉短于约 1 秒的句子（语气词、"嗯""啊"之类），切片后它们大多会被 `min_length` 过滤，或者只贡献噪声。

### 3.3 采样率

- 【事实】上游训练只给 40k / 48k，默认 40k；Applio 的 HiFi-GAN 有 32/40/48k，默认 40k。采样率必须和底模一致，训练中途不能改，要改只能换实验名重新训（FAQ Q18；Applio 训练文档）。
- 【事实】Applio 文档：把有损源转成 WAV 不能恢复已经丢掉的信息（"Converting a lossy source into WAV does not restore lost information"）。
- 【推测】源素材是 44.1/48k 的 OGG（有损），最终产物是有声书（多半还会压成 AAC/MP3），**选 40k 就够了**。40k 是两个实现的默认值，被验证得最多。48k 只多出 20~24 kHz 那一段频带，对人声有声书基本听不出差别，训练和推理却更慢。

### 3.4 要不要剔除极端情绪句

- 【事实】Applio 文档建议数据里包含"natural variation in pitch, articulation, and emotion"，并且说呼吸、清辅音是有用的训练素材，不要默认用门限把它们清掉。文档没有提尖叫和耳语。
- 【事实】RVC 的音高算法在 F0 搜索范围 50~1100 Hz 之外，或者遇到无声段（耳语基本全是清音）时，F0 记为 0（[pipeline.py L74-75](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/81eed5e8f68b6bed1789f682fe78cdd324495afc/infer/vc/pipeline.py#L74-L75)）。
- 【推测】**建议剔除**这几类：尖叫、哭喊、大笑、纯喘息/叹气、耳语/气声、带电话或对讲机滤镜的、带回声混响的（游戏里"心声""回忆"台词经常加混响）、和别人叠声的、带音效的。理由有三：
  1. TTS 输入基本不会出现这些发声方式，训进去只会占模型容量，还会让 index 里混进"非常态"的帧；
  2. 极端发声的 F0 和谐波结构不稳定，容易让模型在正常句子上也出现撕裂或沙哑；
  3. 我们的目标是"统一音色"，不是还原表演。
  
  **保留**情绪中等的句子（开心、生气、低落、害羞等的正常说话），让模型见过一定的音高和力度范围。剔除可以用简单的自动规则先粗筛：句长、RMS 峰均比、有声帧比例、F0 中位数偏离角色均值的程度。

### 3.5 要不要去混响和降噪

- 【事实】Applio 文档：尽量用"little background noise or reverberation"的录音，但不要因为有工具就去混响或做别的处理；降噪只在需要时用，并对比前后，防止辅音细节损失。
- 【事实】上游 2.3 把人声分离后端从 UVR5 换成了 PyMSS（release notes）。
- 【推测】游戏台词是棚录干声，**不需要去混响和降噪**。只把 §3.4 里带效果的那部分句子剔掉就行，不要对整个数据集跑去混响模型，以免引入伪影。

---

## 4. 音高算法与推理参数

### 4.1 音高提取算法

- 【事实】RMVPE（Interspeech 2023，[arXiv:2306.15412](https://arxiv.org/abs/2306.15412)）的测试结果如下：
  - 在 MIR-1K 多声部音乐上 RPA 95.42%，CREPE 91.05%、pYIN 77.29%；
  - MIR-1K 人声加 0 dB 噪声时，平均 RPA 93.60%（Poly 版）/ 92.51%（Vocal 版），CREPE 81.16%。差距主要在 pub 噪声：86.26% 对 61.22%。
- 【事实】FCPE（[arXiv:2509.15140](https://arxiv.org/abs/2509.15140)，[CNChTu/FCPE](https://github.com/CNChTu/FCPE)）：MIR-1K RPA 96.79%，RTX 4090 上 RTF 0.0062。摘要没有直接和 RMVPE、CREPE 比速度。
- 【事实】上游 RVC 和 Applio 的训练、推理默认都是 rmvpe。上游 2.2 的 release notes 说 RMVPE 对低沉男声有改进；上游 2.3 新增了 FCPE 推理。Applio 推理 UI 的说明文字称 rmvpe "is recommended for most cases"。
- 【推测】TTS 输出干净、无伴奏，**训练和推理都用 rmvpe**。FCPE 的优势在实时低延迟，离线批处理用不上。CREPE 更慢，在干净人声上也没有明显优势。个别句子出现音高跳八度或哑音时，再单句换 fcpe 或 crepe 重跑。

### 4.2 推理参数

先看两个实现的默认值，几处不一致：

| 参数 | 含义（源码行为） | 上游 RVC WebUI 默认 | 上游 `infer/cli.py` 默认 | Applio UI 默认 | Applio CLI 默认 |
|---|---|---|---|---|---|
| `index_rate`（Applio 叫 Search Feature Ratio） | 输出特征 = index_rate × 检索特征（top-8 近邻按 1/距离² 加权）+ (1−index_rate) × 原特征 | 0.75 | 0.75 | 0.75 | 0.3 |
| `protect` | 在 F0=0 的清音和呼吸帧上，以 protect 的比例使用检索后的特征，其余保留原始 HuBERT 特征；**=0.5 时整个功能关闭**，值越小保护越强 | 0.33 | 0.33 | 0.5（即关闭） | 0.33 |
| `rms_mix_rate`（Applio 叫 Volume Envelope） | 输出响度包络与输入响度包络的混合比例；越接近 1 越保留模型输出自己的包络，**=1 等于不处理** | 0.25 | 1.0 | 1 | 1.0 |
| `filter_radius` | 旧版只在 harvest 算法下生效，且 >2 时才做一次 3 点中值滤波；**上游 2.3 删除了 harvest，这个参数也随之去掉**；Applio 不对用户暴露 | — | — | — | — |
| `transpose` / `pitch` | F0 × 2^(半音数/12) | 0 | 0 | 0 | 0 |

出处：
- 上游：[webui.py L1869-1921](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/81eed5e8f68b6bed1789f682fe78cdd324495afc/webui.py#L1869-L1921)、[infer/cli.py L57-68](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/81eed5e8f68b6bed1789f682fe78cdd324495afc/infer/cli.py#L57-L68)、[infer/vc/pipeline.py L130、L175-211、L397](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/81eed5e8f68b6bed1789f682fe78cdd324495afc/infer/vc/pipeline.py)；
- Applio：[tabs/inference/inference.py L655-735](https://github.com/IAHispano/Applio/blob/6f69726b140c615c8cc6831a7db8b7c5e48ba8af/tabs/inference/inference.py#L655-L735)、[rvc/infer/pipeline.py L383-416](https://github.com/IAHispano/Applio/blob/6f69726b140c615c8cc6831a7db8b7c5e48ba8af/rvc/infer/pipeline.py#L383-L416)、[core.py L700-830](https://github.com/IAHispano/Applio/blob/6f69726b140c615c8cc6831a7db8b7c5e48ba8af/core.py#L700-L830)；
- filter_radius 的旧行为：[v2.2.231006 pipeline.py L116-120](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/2.2.231006/infer/modules/vc/pipeline.py#L116-L120)。

其他事实：
- 【事实】Applio UI 里 protect 的说明文字写的是"拉满 0.5 提供全面保护"，和代码相反。代码是 `if protect < 0.5:` 才启用。以代码为准，文档页也说 0.5 是关闭。
- 【事实】FAQ Q11 对 index_rate 的说明：index_rate 用来抑制"音色泄露"。设为 1 时理论上消除来自源音频的泄露，但输出会偏向训练集；训练集质量比源音频差时，调高反而降低音质。

---

## 5. 同性别 / 异性别移调

- 【事实】上游 UI 的标签写的是"变调(整数, 半音数量, 升八度12降八度-12)"。Seed-VC 的评测里，男转女 +12、女转男 −12，其他情况 0（[EVAL.md](https://github.com/Plachtaa/seed-vc/blob/main/EVAL.md)）。
- 【事实】移调只改 F0（`f0 *= 2^(key/12)`），音色和共振峰由模型决定。Applio 另外提供 formant shifting 选项，默认关闭。
- 【事实】Applio 的 `proposed_pitch` 功能会算输入 F0 的中位数，然后自动加 `round(12·log2(目标/中位数))` 个半音。目标默认 155 Hz（男）或 255 Hz（女），结果有上下限（[rvc/infer/pipeline.py L213-306](https://github.com/IAHispano/Applio/blob/6f69726b140c615c8cc6831a7db8b7c5e48ba8af/rvc/infer/pipeline.py#L213-L306)）。
- 【推测】我们的 TTS 输入已经是用该角色参考音克隆出来的**同性别**声音，**默认移调 0**。只在 TTS 系统性偏高或偏低时，才按"角色游戏录音 F0 中位数 / TTS 输出 F0 中位数"算整数半音来补偿。旁白之类需要异性别转换时，起点用 ±12，再按中位数比例微调，一般在 ±10~14 之间。
- 【推测】**重要的局限**：RVC 只统一音色（频谱包络），**不改 F0 轮廓**，这点是保留中文声调的前提。所以如果 TTS 的"漂移"里也包括逐句音高整体偏移，RVC 解决不了这部分。可以借鉴 proposed_pitch 的思路，逐句算"把该句 F0 中位数拉向角色中位数"的半音数，限制在 ±2~3 半音内，再作为该句的 transpose 传进去。这一步要实测，太激进会把情绪起伏抹平。

---

## 6. 对 TTS 合成音做转换：情绪、清晰度、伪影的已知数据

### 6.1 有量化数据的

1. 【事实】**Seed-VC EVAL.md 的歌声转换**（[EVAL.md](https://github.com/Plachtaa/seed-vc/blob/main/EVAL.md)）：
   - 设置：源音频是 M4Singer **中文**歌声，每种声部 100 句；每个目标角色各训一个 RVCv2-f0-48k 模型（没给轮数和数据量）；CER 用 SenseVoiceSmall 计算。
   - 总体结果：RVCv2 的 CER 为 28.46，Seed-VC 为 19.70。各声部源音频（Ground Truth）的 CER 平均是 12.66（8.16 / 8.62 / 27.92 / 5.94）。也就是说，**RVC 转换后中文 CER 涨到原来的两倍多**。RVCv2 的 F0 相关度是 0.9404，音质 DNSMOS OVRL 3.12，比 Seed-VC 的 3.06 略好。
   - 注意：这是歌声，比说话难，不能直接套到语音上。
2. 【事实 + 计算】同一张表里还有个跨语言信号。4 个目标角色中，Azuma（东雪莲）、Diana（嘉然）、Ding Zhen（丁真）的训练数据是中文；Kobe Bryant 的数据在 [XzJosh/audiodataset](https://huggingface.co/datasets/XzJosh/audiodataset) 的标注里标为 `EN`，即英语说话。我按表里的数逐格平均：
   - RVCv2：中文目标平均 CER **23.75**，Kobe（英语训练）平均 **42.59**；
   - Seed-VC：中文目标 17.57，Kobe 26.09。
   
   【推测】**用非中文数据训练的目标模型去转中文，清晰度会明显变差**。但这里混杂了"男声、说话数据、数据质量"等因素，不能全归到语言上。
3. 【事实】**Seed-VC EVAL.md 的说话转换（So-VITS-4.0，同属 ContentVec + VITS 一族）**：
   - 设置：源音频是 LibriTTS **英语**，目标是 3 个用日语配音角色训出来的公开 So-VITS 模型（Tokai Teio 等）。
   - 结果：WER 从 6.43 涨到 21.46、48.43、16.25（三个角色分别对应）。
   - EVAL.md 自己说明，结果取决于这些公开模型的质量。
   - 【推测】这是"日语角色模型转其他语言"的旁证：WER 涨了 2.5~7 倍，但模型质量未知。
4. 【事实】**LLVC 论文里的 RVC 对照**（[arXiv:2311.00873](https://arxiv.org/html/2311.00873)）：
   - 设置：RVC v2 32k no-F0 底模，在 LibriSpeech 单个说话人的 39 分钟数据上微调 300 轮。
   - 结果：MOS 自然度 3.58（真人 3.70），相似度 3.35（真人 3.88）；Resemblyzer 相似度 0.846（真人 0.898）；WVMOS 2.465（真人 3.854）。
   - 【推测】这说明 RVC 在说话上能保住大部分自然度，但客观音质分会明显下降。no-F0 版本不是我们会用的配置。
5. 【事实】**英语训练的 SSL 内容特征用在其他语言上**（Amazon，[Investigating self-supervised features for expressive, multilingual VC](https://cdn.amazon.science/90/51/a4ca07c74deba93dc4e60b8fb3db/investigating-self-supervised-features-for-expressive-multilingual-voice-conversion.pdf)）：
   - 设置：HuBERT base 只用英语训练，作为内容特征，声码器用 8 种欧洲语言训练。
   - 结果：WER 只小幅上升，例如英语 7.7% → 8.4%，西语 5.4% → 6.1%，波兰语 5.6% → 6.5%。论文结论是英语训练的 SSL 特征"can successfully generalise to other locales"。
   - 不过这里没有中文、日语这类声调或音节结构差异大的语言，也不是 RVC 本身。

### 6.2 没有找到量化数据的

- 【事实】没有找到用 RVC 转换 TTS 合成语音时，情绪保留或 WER 变化的一手量化数据。上游 wiki 的[对照实验记录](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/%E5%AF%B9%E7%85%A7%E5%AE%9E%E9%AA%8C%C2%B7%E5%AE%9E%E9%AA%8C%E8%AE%B0%E5%BD%95)只有主观听感。其中一条结论是：检索从 top-1 改成 top-8 加权后，"显著削弱高频刺耳的现象"。
- 【推测】从机制看情绪保留：
  - F0 轮廓 100% 来自源；
  - 响度包络在 rms_mix_rate<1 时按比例跟随源；
  - 节奏和时长完全来自源（RVC 是逐帧转换，不改时长）。
  
  所以 TTS 里的**韵律型情绪**（语调起伏、重音、停顿、语速）基本都能保留。**音质型情绪**（气声、紧喉、哭腔、沙哑）会被模型"拉回"训练集的常态发声，index_rate 越高拉得越狠。这也是 §3.4 建议剔除极端句的另一面：剔得越干净，转换出来越"平稳"。

### 6.3 常见伪影与对应参数（来自源码含义和官方说明）

- 清辅音或呼吸出现电音、撕裂：调低 `protect`（0.33 → 0.2）。【事实】上游标签原文："保护清辅音和呼吸声，防止电音撕裂等artifact……调低加大保护力度但可能降低索引效果"。
- 高频刺耳、带上训练集的口癖：调低 `index_rate`。【事实】Applio UI 说明写着"opting for lower values can help mitigate artifacts"。
- 音高跳八度、个别字哑掉：换 F0 算法（rmvpe → fcpe/crepe）重跑该句。【推测】
- 响度忽大忽小：调 `rms_mix_rate`。越接近 0 越跟随 TTS 原本的响度。【事实，见上表】

### 6.4 怎么验证（建议的自测指标）

- 【推测】每个角色留 50~100 句没参与训练的 TTS 句子做验证集，每个 checkpoint 和参数组合都算以下三项：
  1. **CER**：用中文 ASR（如 SenseVoice/Paraformer）识别，对照 TTS 原文本，和"不转换"的 CER 比较；
  2. **说话人相似度**：转换结果和角色游戏原声的 embedding 余弦相似度（Resemblyzer 或 WavLM-SV）；
  3. **逐句相似度的方差**：这才是"漂移"的直接指标。
  
  Applio 训练文档也强调"A lower total loss alone does not prove better voice quality"，要用固定的留出样本比较各个 checkpoint。

---

## 7. 用日语录音训练、转换中文是否有问题

- 【事实】RVC 的内容编码器 `hubert_base` 是 ContentVec。上游 README 的参考项目列表里有 ContentVec，wiki 的对照实验也写作"hubert_base (ContentVec)"。ContentVec 论文的 teacher 数据是**英语 LibriSpeech-960h**：论文写明用 HuBERT base 生成 teacher label，并把"the entire Librispeech-960h"转换成同一个目标说话人后再训练（[arXiv:2204.09224 §4.1](https://arxiv.org/abs/2204.09224)）。也就是说，**内容编码器既没见过日语，也没见过中文**，谈不上"对日语专门优化、对中文不利"。
- 【事实】RVC 官方底模用约 50 小时英语 VCTK 训练（上游 README 原话："底模使用接近50小时的开源高质量VCTK训练集训练"）。
- 【事实】训练和推理必须用同一个 embedder（Applio 训练与推理文档）。Applio 提供 `chinese-hubert-base` 和 `japanese-hubert-base` 作为可选 embedder，权重放在 Applio 自己的 HF 仓库 `IAHispano/Applio/Resources/embedders/`（[rvc/lib/utils.py L193-212](https://github.com/IAHispano/Applio/blob/6f69726b140c615c8cc6831a7db8b7c5e48ba8af/rvc/lib/utils.py#L193-L212)）。上游来源大概率是 TencentGameMate 和 rinna 的同名模型，未核实。
- 【事实】F0 完全取自输入，所以**普通话声调由 TTS 决定，不受训练语言影响**（只要不开 autotune）。
- 【推测】风险点在 **index 检索**。index 里存的全是日语帧的特征。index_rate 高时，每帧中文特征会被拉向最近的 8 个日语帧，日语里没有的音（卷舌 zh/ch/sh/r、前后鼻音 -n/-ng 的区分、ü、儿化、送气对立）可能被"日语化"，听起来就像日本声优在念中文。§6.1 第 2 条里，英语训练的目标模型中文 CER 明显更高，也和这个推测一致。因此：
  1. 跨语言时 **index_rate 从低值起步**（0.3 左右），扫 0 / 0.3 / 0.5 / 0.75，按 CER 和相似度来选；
  2. FAQ 说数据量大、轮数高时 index 作用变小，我们数据多，可以先试 index_rate=0 的效果；
  3. 生成器（VITS 解码器）本身也只学过该角色的日语发音分布。好在内容特征是语言无关的帧级表示，所以生成器多半仍能发出中文音，只是边缘音素的音质可能差一些。
- 【推测】embedder 选择：
  - **基线用 contentvec**。官方和 Applio 的默认底模都是在它上面训的，最稳。
  - chinese-hubert-base 对中文音素的区分可能更好，但训练数据是日语、推理是中文，而且默认 G/D 底模是否针对它训练过不清楚。Applio 文档只要求"compatible pair"。所以它只作为第二轮对照实验。
  - 不要用 japanese-hubert-base，它对中文输入是域外的。

---

## 8. 我们该怎么做（建议）

**实现**
1. 在 Windows 5090 笔记本上装 **Applio 源码版**：`run-install.bat`，Python 3.12，cu128，torch 2.11。装好后先验证 `torch.cuda.get_device_capability()==(12,0)`。备选是上游 `RVC20260718Nvidia50x0.7z`。
2. 一律用 **RVC 架构 + HiFi-GAN + 40000 Hz + contentvec + rmvpe**，保证模型能在上游 RVC 里通用。用 `core.py` 的 CLI 写批处理脚本，不走 UI。

**数据（每个角色）**
3. 从游戏台词里自动粗筛：
   - 去掉短于 1 秒、长于 15 秒的；
   - 去掉有声帧比例过低（耳语、喘息）、F0 中位数偏离角色均值 ±5 半音以上（尖叫、刻意变声）、峰值削波的；
   - 去掉文件名或脚本标注为心声、回忆、电话等带效果的。
   
   然后人工抽听确认。
4. 第一版取 **约 45 分钟**（几百到一千多句）、情绪平稳到中等的台词；第二版取 **约 2~3 小时**做对照。不去混响，不降噪，归一化用 Applio 默认的 post。预处理用 `Automatic`（3 秒、0.3 秒重叠）。留出 50~100 句做验证。

**训练**
5. batch size 先取 8~16，看显存决定，笔记本 5090 是 24 GB。45 分钟那版 `total_epoch` 设 200~300、`save_every_epoch` 设 10、开 `save_every_weights`；2~3 小时那版轮数按总步数相当来折算，大约 60~100。用 §6.4 的三项指标在留出集上挑 checkpoint，不看 loss。
   - 【推测】这些轮数是根据 FAQ"干净数据 200 是 ok 的"推出来的起点，没有一手数据支撑最佳值。

**推理参数起点**（TTS 中文 → 日语训练的角色模型）

| 参数 | 起点 | 扫描范围 |
|---|---|---|
| f0 method | rmvpe | 个别坏句换 fcpe |
| transpose | 0（同性别）/ ±12（异性别） | 按 F0 中位数比例取整，必要时逐句 ±2 以内校正 |
| index_rate | **0.3** | 0 / 0.3 / 0.5 / 0.75 |
| protect | **0.33** | 0.2~0.4；注意 0.5 是关闭 |
| rms_mix_rate（volume envelope） | **0.25~0.5** | 让响度跟随 TTS，以保留情绪起伏；1 = 不处理 |
| filter_radius | 不适用 | rmvpe 和 fcpe 都不受它影响；只有旧版 harvest 时取 3 |
| autotune / formant shift / clean audio | 全关 | — |

**验收**
6. 先拿雪菜（8600 句）做试点。比较三组：不转换、RVC（index 0.3）、RVC（index 0）。看 CER 变化、相似度均值、逐句相似度方差。CER 上升控制在可接受范围（例如绝对值 +2 个百分点以内，阈值由我们自定），同时方差明显下降，才推广到其他角色。如果 CER 涨得厉害，先降 index_rate，再考虑换 chinese-hubert-base，最后才考虑换 Seed-VC 类的零样本方案。

---

## 参考来源

- RVC-Project WebUI：[仓库 @81eed5e](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/tree/81eed5e8f68b6bed1789f682fe78cdd324495afc)、[Releases](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/releases)、[2.3.260718](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/releases/tag/2.3.260718)、[docs/cn/faq.md](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/81eed5e8f68b6bed1789f682fe78cdd324495afc/docs/cn/faq.md)、[docs/cn/cli.md](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/81eed5e8f68b6bed1789f682fe78cdd324495afc/docs/cn/cli.md)、[wiki 对照实验记录](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/%E5%AF%B9%E7%85%A7%E5%AE%9E%E9%AA%8C%C2%B7%E5%AE%9E%E9%AA%8C%E8%AE%B0%E5%BD%95)、[HF 模型仓库 lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI)
- Applio：[仓库 @6f69726](https://github.com/IAHispano/Applio/tree/6f69726b140c615c8cc6831a7db8b7c5e48ba8af)、[文档首页](https://docs.applio.org/)、[安装](https://docs.applio.org/getting-started/installation/)、[训练](https://docs.applio.org/getting-started/training/)、[推理](https://docs.applio.org/getting-started/inference/)、[数据集](https://docs.applio.org/guides/how-create-datasets/)
- PyTorch 2.7 Blackwell / cu128：<https://pytorch.org/blog/pytorch-2-7/>
- RMVPE：<https://arxiv.org/abs/2306.15412>
- FCPE：<https://arxiv.org/abs/2509.15140>，<https://github.com/CNChTu/FCPE>
- ContentVec：<https://arxiv.org/abs/2204.09224>，<https://github.com/auspicious3000/contentvec>
- Seed-VC 评测：<https://github.com/Plachtaa/seed-vc/blob/main/EVAL.md>，论文 <https://arxiv.org/abs/2411.09943>，目标角色数据 <https://huggingface.co/datasets/XzJosh/audiodataset>
- LLVC（含 RVC 对照）：<https://arxiv.org/abs/2311.00873>
- 英语 SSL 特征的跨语言 VC：<https://cdn.amazon.science/90/51/a4ca07c74deba93dc4e60b8fb3db/investigating-self-supervised-features-for-expressive-multilingual-voice-conversion.pdf>
