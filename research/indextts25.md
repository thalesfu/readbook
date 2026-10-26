# IndexTTS-2.5 在 RTX 5090 笔记本上怎么用：音色固定、逐句换情绪

- 对应票：`.scratch/voice-pipeline/issues/02-indextts25-research.md`
- 调研日期：2026-10-26
- 源码快照：`index-tts/index-tts` main @ [`d9e41aa`](https://github.com/index-tts/index-tts/tree/d9e41aac89fd00b3d71497fddb287b7f24613712)（2026-09-30）。下文代码链接都钉在这个提交上，记作 `@d9e41aa`。
- 标注约定：**【事实】** 有一手来源（源码、官方 README、模型卡、论文、维护者回复）；**【社区】** 来自 issue 里的用户报告，没有官方确认；**【推测】** 是我的推断或建议，需要上机验证。

---

## 0. 结论速览

1. **【事实】** 2.5 的权重和推理代码都已公开（2026-08-10 发布）：代码在 GitHub `index-tts/index-tts` 的 `indextts/infer_v2_5.py`，权重在 HuggingFace / ModelScope 的 `IndexTeam/IndexTTS-2.5`。接口和 2.0 基本一致，区别有五处：多了必填参数 `lang`；多了 `duration_factor`；半精度开关从 `use_fp16` 改成 `use_bf16`；`use_qwen_emo` 默认改为关；GPT 的说话人条件改用 CAMPPlus 全局向量。
2. **【事实】** 8 维情感向量**不是**直接调一个情绪旋钮。代码按“与参考音色最相似的说话人”从一个固定的情感样本库（2.0 与 2.5 共用同一份）里挑示例嵌入，再按权重加权；剩下的 `1 − Σw` 留给定妆音自身的情绪。所以**权重总和越大，定妆音自身的成分越少，音色泄漏风险越高**。官方 WebUI 会把总和压到 ≤0.8 并降低“平静/惊讶”的权重，**Python API 不会**，要自己调用 `tts.normalize_emo_vec()`。
3. **【事实】** 定妆音超过 15 秒的部分会被直接截掉（代码只取前 15 秒）。10–18 秒的定妆音最好自己裁到 ≤15 秒，保留最干净的一段。
4. **【事实+社区】** 在 Windows 上只装 `webui` extra 最稳妥，DeepSpeed、flash-attn（`accel`）在 Windows 上都难装。官方 RTF 数据：2.5 bf16 在 RTX 4090 上约 0.20。**【推测】** 5090 笔记本上整句平均 RTF 在 0.25–0.4 之间，短句更慢。有用户报告 2.5 的**中文**自然度和情感表现不如 2.0（#759/#781/#801），建议上机后两版 A/B。
5. **【推测】** 推荐做法：情绪用向量控制（我们的剧本已经有标签，没必要再让 Qwen 去猜）；“平常”不传向量；向量总和控制在 0.55–0.8；`emo_alpha` 当作全书或单个角色的强度旋钮，取 0.7–1.0。“害羞”和“轻声”在 8 维里没有对应维度，改用**同一角色自己的**情绪参考音频（`emo_audio_prompt`）更合适。每句生成后算一次说话人相似度，低于阈值就重生成。

---

## 1. 发布状态、权重位置与 2.0 的接口差异

### 1.1 发布状态 【事实】

| 项 | 内容 | 来源 |
|---|---|---|
| 发布时间 | 2026/08/10，README News：“We release IndexTTS-2.5” | [README @d9e41aa](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/README.md) |
| 推理代码 | `indextts/infer_v2_5.py`（类名仍为 `IndexTTS2`），WebUI 默认就是 2.5（`--version 2.5`） | [infer_v2_5.py](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/indextts/infer_v2_5.py)、[webui.py#L28](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/webui.py#L28) |
| HF 权重 | [huggingface.co/IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5)：创建于 2026-08-10，非 gated。`gpt.pth` 3.26 GB，`s2mel.pth` 415 MB，`codec.pth` 607 MB，另有 `qwen0.6bemo4-merge/`（情感文本模型）、`feat1.pt`、`feat2.pt`（情感样本库） | HF API `/api/models/IndexTeam/IndexTTS-2.5/tree/main` |
| ModelScope | [modelscope.cn/models/IndexTeam/IndexTTS-2.5](https://modelscope.cn/models/IndexTeam/IndexTTS-2.5) | README Model Zoo |
| 在线试用 | [HF Space](https://huggingface.co/spaces/IndexTeam/IndexTTS-2.5-Demo)、[ModelScope Studio](https://modelscope.cn/studios/IndexTeam/IndexTTS-2.5) | README Model Zoo |
| 论文 | IndexTTS 2.5 Technical Report，[arXiv:2601.03888](https://arxiv.org/abs/2601.03888)（2026-01） | — |
| 许可 | bilibili Model Use License Agreement（非开源许可；商用需联系 indexspeech@bilibili.com）。个人自用没有问题 | [LICENSE](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/LICENSE)、模型卡 |
| 模型卡写的要求 | Python 3.10–3.11，NVIDIA GPU，推理约 6 GB 显存 | [模型卡 README](https://huggingface.co/IndexTeam/IndexTTS-2.5) |

**【事实】** 2.0 和 2.5 的 `feat1.pt`、`feat2.pt` 以及 `qwen0.6bemo4-merge/model.safetensors` 的 LFS 哈希完全相同（`f219cb44…`、`9c4292e9…`、`11293257…`），`config.yaml` 里的 `emo_num: [3, 17, 2, 8, 4, 5, 10, 24]` 也一致。也就是说，**情感样本库和文本→情感模型两版共用**，下面的情绪映射表两版都能用；GPT 与 s2mel 的权重则不同。

### 1.2 论文里的改动 【事实】（[arXiv:2601.03888](https://arxiv.org/html/2601.03888v3)）

- 语义 token 码率从 50 Hz 降到 25 Hz，序列长度减半（§3.2）；TRT 后端 README 实测 50.1 → 25.1 tokens/秒音频。
- 论文称 S2M 用 Zipformer 替换了 U-DiT（§3.2），**但公开的 `config.yaml` 仍是 `dit_type: "DiT"`**。issue [#801](https://github.com/index-tts/index-tts/issues/801) 也问过公开的 `s2mel.pth` 到底是哪个版本，截至快照没有官方答复。
- T2S 做了 GRPO 后训练（以 WER 为奖励），产出 2.5-RL 变体（§3.3）。公开的只有一份 `gpt.pth`，是不是 RL 版没有说明（#801）。
- 情感训练数据仍是 135 小时（ESD 29 小时 + 商业数据 106 小时，§4.1），和 2.0 一样。
- 2.5 论文的情感评测只做了日语、西语对比 CosyVoice 3（表 3），**没有中文情感评测，也没有讨论音色泄漏**。

### 1.3 Python API 差异 【事实】

| | IndexTTS-2.0 | IndexTTS-2.5 |
|---|---|---|
| 导入 | `from indextts.infer_v2 import IndexTTS2` | `from indextts.infer_v2_5 import IndexTTS2` |
| 权重目录（README 约定） | `checkpoints_2/` | `checkpoints/` |
| 半精度 | `use_fp16=True` | `use_bf16=True`（**没有 FP16**：维护者在 [PR #798](https://github.com/index-tts/index-tts/pull/798) 说 FP16 推理在内部评测中有问题，拒绝合并 FP16 回退） |
| `lang` | 无 | **必填**（`"ZH"`/`"EN"`/`"JA"`/`"ES"`/`"AR"`；代码里还认 `zhen`） |
| `duration_factor` | 无 | 0.5–2.0，>1 变慢 |
| QwenEmotion | 构造参数 `use_qwen_emo` 默认 `True` | 默认 `False`；不开就用 `use_emo_text=True` 会抛 `RuntimeError` |
| GPT 说话人条件 | Conformer/Perceiver 处理序列特征 | `spk_cond_mode="campplus"`：192 维 CAMPPlus 全局向量经 `spk_emb_proj` 投影（[infer_v2_5.py#L139](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/indextts/infer_v2_5.py#L139)、[model_v2_5.py#L349](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/indextts/gpt/model_v2_5.py#L349)） |
| `use_gpt_latent` | 有 | 已删除（[#767](https://github.com/index-tts/index-tts/pull/767)） |
| 发音标注 | 混写拼音 `XING2` | `<行|XING2>` 形式，可以写拼音、CMU 音素、日文假名 |
| 其余参数 | `spk_audio_prompt, text, output_path, emo_audio_prompt, emo_alpha, emo_vector, use_emo_text, emo_text, use_random, interval_silence, max_text_tokens_per_segment, stream_return` 与采样参数（`do_sample/top_p/top_k/temperature/num_beams/repetition_penalty/max_mel_tokens`）两版一致 | 同左 |

2.5 构造函数签名（[infer_v2_5.py#L77-L80](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/indextts/infer_v2_5.py#L77-L80)）：

```python
IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_bf16=False, device=None,
          use_cuda_kernel=None, use_deepspeed=False, use_accel=False, use_torch_compile=False, use_qwen_emo=False)
```

注意：`docs/README2.5_ZH.md` 里的路径（`checkpoints_25/config_v2_5.yaml`）已经过时，**以英文 README 为准**。

---

## 2. 情感控制在代码里怎么工作（决定了怎么调参）【事实】

依据 [infer_v2_5.py `infer_generator` L570–L790](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/indextts/infer_v2_5.py#L570-L790) 和 [model_v2_5.py `merge_emovec` L792–L797](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/indextts/gpt/model_v2_5.py#L792-L797)：

1. **四种模式互斥**：只要给了 `emo_vector` 或 `use_emo_text=True`，`emo_audio_prompt` 就会被清掉（L583–L586）。
2. **文字模式**：`use_emo_text=True` 时，用 `emo_text`（没给就用正文 `text`）调用 QwenEmotion，得到 8 维字典，再转成 `emo_vector`（L588–L600）。所以**文字模式最后也是变成向量**，只是多了一步模型猜测。
3. **向量模式里的 `emo_alpha`**：把向量整体乘以 `clamp(emo_alpha, 0, 1)`，保留 4 位小数（L602–L610）。之后内部的 `emo_alpha` 被强制改为 1.0（L612–L617）。
4. **样本库选取**：`emo_matrix`（`feat2.pt`）按 `emo_num=[3,17,2,8,4,5,10,24]` 切成 8 组，依次是喜 3、怒 17、哀 2、惧 8、厌 4、忧郁 5、惊讶 10、平静 24 个示例；`spk_matrix`（`feat1.pt`）是这些示例对应的说话人向量。`use_random=False` 时，每组挑出说话人向量与定妆音 CAMPPlus 向量**余弦相似度最高**的那个示例（L665–L675，[`find_most_similar_cosine` L901](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/indextts/infer_v2_5.py#L901)）；`use_random=True` 时随机挑选，README 明说这样会降低音色还原度。
5. **混合公式**（L764）：

   ```
   emovec = Σ_i w_i · E_i(挑中的示例)  +  (1 − Σ_i w_i) · emovec(定妆音自身)
   ```

   **Σw = 0** 时（不传向量，或全零），情绪完全来自定妆音；**Σw → 1** 时，定妆音自身的情绪成分趋近 0；**Σw > 1** 时，定妆音成分变成负权重（代码不拦截，属于未定义行为）。
6. **官方归一化只在 WebUI 里做**：[`normalize_emo_vec` L489–L503](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/indextts/infer_v2_5.py#L489-L503) 先乘以偏置 `[0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]`（注释：降低“容易产生怪异结果”的情绪），再把总和压到 ≤0.8。只有 [webui.py#L678](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/webui.py#L678) 调用了它，**`tts.infer()` 不会自动调用**。
7. **情感参考音频模式**：`merge_emovec` 是线性插值 `base + α·(emo − base)`，其中 base 是定妆音的情感嵌入，emo 是情感参考音频的嵌入。`emo_alpha` 就是插值系数。
8. **缓存**：定妆音特征按文件路径缓存，只缓存**一个**说话人（L620）。情感参考音频同样只缓存一个。向量模式下情感参考就是定妆音本身，**逐句换向量不会重新提取参考特征**。
9. **定妆音截断**：`_load_and_cut_audio(spk_audio_prompt, 15)`，超过 15 秒只取前 15 秒（[L397–L409](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/indextts/infer_v2_5.py#L397-L409)、[L627](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/indextts/infer_v2_5.py#L627)）。情感参考音频也一样。
10. **QwenEmotion**（[L909 起](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/indextts/infer_v2_5.py#L909)）：中文标签 高兴/愤怒/悲伤/恐惧/反感/低落/惊讶/自然 对应 8 维。每维限幅到 [0, 1.2]；全为 0 时回落到 calm=1.0。代码注释承认它**分不清“悲伤”和“低落”**，只有文本里出现“低落/melancholy/depressed/gloomy”等词时才会把两者对调。曾有因输出非数值而报错的情况（[#736](https://github.com/index-tts/index-tts/issues/736)，已关闭；[#751](https://github.com/index-tts/index-tts/pull/751) 做了标签式输出的兼容）。
11. 论文口径：IndexTTS2 论文的 T2E 只定义了 **7 类**（怒、喜、惧、厌、哀、惊、中性，§3.3），蒸馏自 DeepSeek-R1，用约 1000 对样本训练 Qwen3-1.7B LoRA。代码里实际是 8 维（多了 melancholic），模型是 0.6B（目录名 `qwen0.6bemo4-merge`）。**论文和实现不一致，以代码为准。**（[arXiv:2506.21619](https://arxiv.org/html/2506.21619)）

---

## 3. Windows + RTX 5090 笔记本（24 GB，Blackwell sm_120，CUDA 12.8）安装要点

### 3.1 依赖事实 【事实】（[pyproject.toml @d9e41aa](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/pyproject.toml)）

- `requires-python = ">=3.10,<3.12"`；uv 是**必需**的（README 原话“required”），`uv sync` 会自动装好 Python。
- Windows/Linux 上 `torch==2.8.*` / `torchaudio==2.8.*` 来自 `download.pytorch.org/whl/cu128`。PyTorch 从 2.7.0 起，cu128 构建就支持 Blackwell，包括 sm_120（[PyTorch 论坛维护者说明](https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099)），所以**不需要自己编译 torch**。
- extras：`webui`（gradio 5.45）、`deepspeed`（0.17.1）、`accel`（flash-attn 2.8.3.post1 + Windows 上的 `triton-windows==3.1.0.post17`）、`torch_compile`（Windows 上的 triton-windows）。
- `uv.lock` 里 flash-attn **只有 sdist、没有 wheel**，在 Windows 上要现场编译。
- README：Windows 上 DeepSpeed“may be difficult to install”，建议不用 `--all-extras`，手动挑 extra；安装时报 CUDA 错误就装 CUDA Toolkit ≥12.8。
- 24 GB 显存远高于 WebUI 的低显存阈值 10 GB（[webui.py#L115-L130](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/webui.py#L115-L130)），不会触发自动半精度和自动分段，QwenEmotion 也会照常加载。

### 3.2 建议步骤 【推测，基于上面的事实】

```powershell
# 1. 驱动：支持 CUDA 12.8 的 NVIDIA 驱动（R570 及以上）。只用预编译 wheel 的话，不需要装 CUDA Toolkit
git clone https://github.com/index-tts/index-tts.git; cd index-tts
pip install -U uv
uv sync --extra webui          # 先别用 --all-extras（会去编译 DeepSpeed 和 flash-attn）
uv tool install "huggingface-hub"
hf download IndexTeam/IndexTTS-2.5 --local-dir=checkpoints
hf download IndexTeam/IndexTTS-2   --local-dir=checkpoints_2   # 留着做 A/B
uv run tools/gpu_check.py      # 确认 torch 能看到 5090
uv run webui.py --fp16         # 2.5 下 --fp16 实际开的是 BF16（webui.py#L155-L159）
```

Python 里：

```python
from indextts.infer_v2_5 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints",
                use_bf16=True, use_cuda_kernel=False, use_qwen_emo=False)
```

### 3.3 各加速选项的取舍

| 选项 | 事实 | 对我们的建议【推测】 |
|---|---|---|
| `use_bf16` | 官方推荐 2.5 用 BF16。README 表：4090 上 2.5 bf16 的 RTF 0.2065，fp32 是 0.2060，**几乎一样** | 开。主要好处是省显存；24 GB 下 fp32 也能跑，两种都试一下听感 |
| FP16 | 2.5 官方不支持（PR #798） | 不用 |
| `use_cuda_kernel`（BigVGAN 融合激活） | 旧代码硬编码 `compute_70`，在 sm_120 上编译失败（[#468](https://github.com/index-tts/index-tts/issues/468)）；2026-08-13 的 [#762](https://github.com/index-tts/index-tts/pull/762) 改为按检测到的显卡架构编译。编译要 nvcc + MSVC。#584 的 5090 日志里 `bigvgan_time` 只有 0.08 秒 | 收益很小，**先关**。编译失败会自动回退到 torch 实现 |
| `use_deepspeed` | README：可能变快，也可能变慢；Windows 难装 | 不用 |
| `use_accel`（flash-attn + CUDA Graph + paged KV 的 GPT2 引擎） | [PR #517](https://github.com/index-tts/index-tts/pull/517) 在 2.0 上报告 GPT 阶段加速 2.4–3.8 倍；[#580](https://github.com/index-tts/index-tts/issues/580) 报告在 A100 上没有加速，在 H100 上有效。2.5 代码里 accel 模型在 `half=True` 时调用 `.half()`，即 **FP16**（[model_v2_5.py#L409-L437](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/indextts/gpt/model_v2_5.py#L409-L437)），这和“2.5 不支持 FP16”的说法相冲突 | GPT 自回归是主要瓶颈，这项最可能带来收益。但 Windows 上要编译 flash-attn，`triton-windows 3.1` 能否支持 sm_120 也存疑；开了 bf16 时还会悄悄走 FP16。**放到第二阶段，最好在 WSL2 里试** |
| `use_torch_compile`（只编译 s2mel） | 需要 triton，Windows 上是 triton-windows | 可以试，收益未知 |
| TensorRT 后端 `backends/trt/` | [PR #733](https://github.com/index-tts/index-tts/pull/733)：**只支持 2.0**，独立 venv（torch 2.7、TRT-LLM 0.21）。同机 RTF：2.0 TRT fp16 0.1365，2.0 PyTorch fp16 0.3263，2.5 PyTorch bf16 0.2035 | 2.5 用不上；如果 A/B 后选了 2.0，可以考虑（Linux/WSL2） |
| vLLM | README 给了 [vLLM recipe](https://recipes.vllm.ai/IndexTeam/IndexTTS-2.5)，面向生产部署 | 个人批量合成用不上；Windows 原生也不支持 |
| audio.cpp（C++/GGML，第三方） | 作者在 [#722](https://github.com/index-tts/index-tts/issues/722) 自报 5090 上 2.5 Q8_0 约 10.8 倍实时、约 5 GB 显存，2.0 F16 约 8.7 倍实时 | 第三方实现，与官方实现的一致性没有独立验证；Python 那条路太慢的话可以作为备选 |

---

## 4. 预期实时率

**【事实】**

- README（RTX 4090，`kv_cache=True`）里 2.5 bf16 的 RTF：7 字 0.287，16 字 0.216，28 字 0.207，80 字 0.200，200 字 0.200。2.0 fp16 整体是 0.326。
- 2.5 论文（A10）：T2S 的 RTF 从 0.232 降到 0.119，摘要称整体提升 2.28 倍。
- [#584](https://github.com/index-tts/index-tts/issues/584)：RTX 5090（Windows 和 WSL2 都试过，看日志是 **2.0** 加情感向量）合成一句 2.84 秒的音频，总耗时 3.29 秒（RTF 1.16），其中 `gpt_gen_time` 2.69 秒，s2mel 0.50 秒，BigVGAN 0.08 秒。说明**短句在 Windows 上可能被 GPT 自回归的单步开销拖住**；这份报告没有写精度和 accel 设置。
- [#722](https://github.com/index-tts/index-tts/issues/722)：同一作者在 5090 上测 Python 版 2.0，长文本加 emotion_text 的 RTF 为 1.84。

**【推测】**

- RTX 5090 笔记本是 GB203、10496 CUDA 核、TGP 95–150 W（[Notebookcheck](https://www.notebookcheck.com/Nvidia-enthuellt-GeForce-RTX-5090-5080-und-5070-Laptop-GPUs-mit-bis-zu-10-496-CUDA-Kernen-und-24-GB-GDDR7.943054.0.html)），算力明显低于桌面 4090/5090。自回归解码对 CPU 单核和 kernel launch 开销也敏感。
- 按逐句合成（中文一句 10–40 字）估计，**2.5 bf16 的 RTF 大约 0.25–0.5，即 2–4 倍实时**。如果遇到 #584 那种情况，短句的 RTF 会接近 1。一本 10 小时的书大约需要 3–5 小时 GPU 时间，过夜批处理可以接受。
- 提速手段（按性价比排序）：
  1. **按角色排序批量合成**，命中定妆音缓存（代码只缓存一个说话人）；
  2. 把同角色、同情绪的相邻短句合成一次再切开（会改变停顿，需评估）；
  3. 去 WSL2 里试 `use_accel`；
  4. 试 audio.cpp。
- 需要实测：首句的预热开销（TRT README 提到第一次迭代慢约 55%）。

---

## 5. 10 类情绪 → 8 维向量建议映射表（**全部为推测**）

向量顺序：`[喜 happy, 怒 angry, 哀 sad, 惧 afraid, 厌恶 disgusted, 忧郁 melancholic, 惊讶 surprised, 平静 calm]`。

设计原则（来自第 2 节的机制）：

- 总和越大，定妆音自身的成分越少，泄漏风险越高。所以**归一化后的总和控制在 0.55–0.8**，强情绪取上限，弱情绪取下限。
- 下表“原始向量”是送进 `tts.normalize_emo_vec(vec, apply_bias=True)` **之前**的值。“归一化后”一列是用官方函数算出的结果（偏置 + 总和 ≤0.8），“定妆音份额”就是 `1 − Σw`。
- 样本库里“哀”只有 2 个示例、“喜”只有 3 个，挑到的示例未必贴合我们的角色，所以“哀”系情绪建议混入“忧郁”（5 个示例）分摊；社区贡献者也提到“忧郁”这一维很能增加人味（[#314](https://github.com/index-tts/index-tts/issues/314)）。

| 我们的情绪 | 原始向量 | 归一化后（官方函数） | 定妆音份额 | 说明 / 信心 |
|---|---|---|---|---|
| 平常 | **不传**（`emo_vector=None`） | — | 1.00 | 定妆音本来就是中性表演，情绪完全取自它，音色最稳。信心高 |
| 开心 | `[0.6, 0, 0, 0, 0, 0, 0, 0.2]` | `[0.562,0,0,0,0,0,0,0.113]` | 0.32 | 喜为主，带一点平静，避免“过嗨”。信心中 |
| 兴奋 | `[0.75, 0, 0, 0, 0, 0, 0.25, 0]` | `[0.643,0,0,0,0,0,0.157,0]` | 0.20 | 喜加惊讶，提高能量。信心中 |
| 生气 | `[0, 0.75, 0, 0, 0.1, 0, 0, 0]` | `[0,0.656,0,0,0.094,0,0,0]` | 0.25 | 怒的示例最多（17 个），较可靠；轻度生气可把怒降到 0.4–0.5。信心中高 |
| 难过 | `[0, 0, 0.45, 0, 0, 0.3, 0, 0]` | `[0,0,0.45,0,0,0.281,0,0]` | 0.27 | 哀加忧郁，偏内敛。信心中 |
| 哭腔 | `[0, 0, 0.8, 0.1, 0, 0.1, 0, 0]` | `[0,0,0.644,0.081,0,0.075,0,0]` | 0.20 | 哀拉满，带一点惧（颤抖感）。**泄漏风险最高**，优先用同角色的哭腔参考音频。信心中低 |
| 害羞 | `[0.3, 0, 0, 0.2, 0, 0, 0, 0.2]` | `[0.281,0,0,0.2,0,0,0,0.113]` | 0.41 | 8 维里没有“羞”，用“轻微喜 + 轻微惧（怯）+ 平静”近似。**信心低**，建议用参考音频 |
| 冷淡 | `[0, 0, 0, 0, 0.2, 0.1, 0, 0.5]` | `[0,0,0,0,0.188,0.094,0,0.281]` | 0.44 | 平静为主，带一点厌恶（疏离）和忧郁（低能量）。平静的偏置只有 0.5625，实际效果偏弱。信心中低 |
| 惊讶 | `[0.1, 0, 0, 0.1, 0, 0, 0.7, 0]` | `[0.094,0,0,0.1,0,0,0.481,0]` | 0.33 | 惊讶的偏置是 0.6875，官方认为它容易出怪音。可按剧本提示偏喜（惊喜）或偏惧（惊吓）。信心中 |
| 轻声 | `[0, 0, 0, 0.1, 0, 0.25, 0, 0.4]` | `[0,0,0,0.1,0,0.234,0,0.225]` | 0.44 | “轻声”是音量/发声方式，不是情绪，向量**基本控制不了气声和音量**。这里只能做到“低能量”的近似。**信心低**，建议用同角色的轻声参考音频，或在后期降增益 |

实现时可以直接调用官方函数：

```python
vec = tts.normalize_emo_vec(raw_vec, apply_bias=True)   # API 不会自动做这一步
tts.infer(spk_audio_prompt=ref, text=line, lang="ZH", output_path=out,
          emo_vector=vec, emo_alpha=book_intensity, use_random=False)
```

---

## 6. `emo_alpha` 怎么设；文字描述与向量的取舍

### 6.1 `emo_alpha` 【事实】

- 范围 0–1，默认 1.0（README；代码里会钳位）。
- 向量模式下，它就是把向量整体缩放，相当于把 Σw 往 0 拉，把份额还给定妆音（第 2 节第 3、5 条）。
- 情感参考音频模式下，它是定妆音情感嵌入和参考音频情感嵌入之间的线性插值系数。README 示例用的是 0.9。
- 文字模式下，README 建议“around 0.6 (or lower)”更自然。**【推测】** 原因是 QwenEmotion 输出的各维之和通常接近 1（论文 T2E 是概率分布），alpha=1 时 `1−Σw≈0`，定妆音的情绪成分会被完全挤掉；alpha=0.6 时大约保留 40%。

### 6.2 建议 【推测】

- 向量模式：**强度直接写进映射表**，`emo_alpha` 只当全局旋钮——全书默认 0.85–1.0；嗓音容易“跑”的角色单独下调到 0.6–0.7；剧本提示里写“强烈/崩溃”的句子不改 alpha，改用映射表的强档。
- 参考音频模式（害羞/轻声/哭腔）：从 0.7–0.9 起试。参考音频是**同一角色**录的，alpha 可以开高；是**别人**的，alpha 不要超过 0.6。

### 6.3 文字描述（`use_emo_text` / `emo_text`）与向量的取舍

| | 向量 `emo_vector` | 文字 `use_emo_text` + `emo_text` |
|---|---|---|
| 可控性 / 可复现 | 确定性，同一标签得到同一向量 | 多一层 0.6B LLM 推断，输出可能波动 |
| 表达范围 | 都限于 8 维 | 一样限于 8 维（最终也变成向量），**不会因为写了“害羞地小声说”就多出维度** |
| 已知缺陷 | 要自己定映射 | 分不清悲伤和低落（代码注释）；曾输出非数值导致报错（#736）；WebUI 把它列为“实验功能”，默认隐藏（[webui.py#L171-L176](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/webui.py#L171-L176)） |
| 资源 | 不需要额外模型 | 要 `use_qwen_emo=True`，多加载一个 0.6B 模型（24 GB 显存无压力） |

**【推测】结论**：我们的剧本已经给每句标好了情绪，主路径用**向量**。剧本里那句“表演提示”可以离线跑一次 QwenEmotion（`tts.qwen_emo.inference(提示)`），**只拿结果来对照、校准映射表**，不直接用于合成。如果某类情绪向量怎么调都不对，再对那一类试 `emo_text=表演提示, emo_alpha≤0.6`。

---

## 7. 音色泄漏：证据与缓解

### 7.1 证据

- **【事实·第三方论文】** DisCo-Speech（[arXiv:2512.13251](https://arxiv.org/abs/2512.13251)，§4.2.2、表 3）拿 IndexTTS 2 做 AB 偏好测试。在情感场景的“音色”维度上，DisCo-Speech 42.5%，无偏好 23.8%，IndexTTS 2 33.7%；“韵律”维度 IndexTTS 2 略胜（38.5% 对 36.0%）。作者原话：“Index-TTS 2's strong emotional expressiveness often comes at the cost of source timbre leakage.” 作者的解释是，强情感本身就会改变音色，IndexTTS2 把两者纠缠在一起。这是竞品作者的评测，有立场，但与社区反馈一致。
- **【社区】** [#292](https://github.com/index-tts/index-tts/issues/292)：选“与音色参考相同”时相似度很高；换成其他情感控制方式后，“几乎成了另一个不相干的声音”，多人附议说“音色更接近情感参考音频”。[#314](https://github.com/index-tts/index-tts/issues/314)：加情绪后 AI 感变重，贡献者建议不要把滑块全拉满。[#611](https://github.com/index-tts/index-tts/issues/611)：情感表现浮夸、重读过多。
- **【社区】2.5 中文退步**：[#759](https://github.com/index-tts/index-tts/issues/759)（“音色相似度、情感色彩不如 2.0”，多人附议）、[#781](https://github.com/index-tts/index-tts/issues/781)（已关闭，无官方解释）、[#801](https://github.com/index-tts/index-tts/issues/801)（单人在 MPS 上 6 组盲测，2.0 胜 5 组；并质疑 CAMPPlus 单向量条件和 25 Hz 是否损失了中文细节）。截至快照，维护者没有回应。
- **【事实·官方论文】** IndexTTS2 论文表 2：情感测试集上 SS 0.836、ES 0.887；去掉 GPT latent 后 SS 升到 0.869，WER 变差。论文**没有**给出“有情感控制 vs 无情感控制”的 SS 对比，也没有讨论泄漏。2.5 论文同样没有。
- **【事实·机制】** 从代码看，泄漏有三个来源：
  - 情感嵌入来自**别的说话人**的示例或参考音频，GRL 解耦并不完美；
  - Σw 越大，定妆音自身的嵌入占比越小；
  - `use_random=True` 时示例不再按说话人相似度挑选。

  最终音色由 s2mel 阶段的定妆音 mel 提示和 CAMPPlus style 注入，但 GPT 生成的语义 token 已经带有韵律和部分音色信息。

### 7.2 缓解办法

**官方或代码里已有的【事实】**

- 保持 `use_random=False`（默认）；
- 向量先过 `normalize_emo_vec`（总和 ≤0.8，弱化平静和惊讶）；
- 情感参考音频模式用 `emo_alpha<1` 插值；
- 文字模式用 `emo_alpha≈0.6`。

**【推测】工程上可以加的**

1. **情绪参考音频尽量来自同一角色**：给每个主要角色多录或多挑几条“哭腔、轻声、害羞、愤怒”的定妆音，用 `emo_audio_prompt` 驱动。这样即使泄漏，泄进来的也是自己的音色。也可以先用向量模式生成一批，挑出音色最稳的一条，固定下来当这一类情绪的参考音频，保证全书一致。
2. **自动质检、重生成**：每句生成后，用 CAMPPlus 或 ECAPA 算一次与定妆音的余弦相似度（IndexTTS 自带的 `campplus_model` 可以直接用）。按“平常”句的分布定阈值，低于阈值就降强度（Σw 乘 0.7）或换种子重生成，最多试 N 次。
3. **强情绪降档**：剧本标“生气/哭腔”的句子默认用中档；表演提示里明确写“爆发、崩溃”时才用强档。
4. **定妆音质量**：干声、无混响和底噪、情绪中性、≤15 秒，自己裁剪，不依赖截断。#759 里也有人强调参考音频的质量影响很大。
5. **后处理兜底**：对泄漏严重又不想重录的句子，用 seed-vc 一类的 VC 把音色拉回定妆音（IndexTTS2 的 s2mel 本身就借鉴了 seed-vc，见 README 致谢）。会增加耗时，也有失真风险，只作最后手段。
6. **2.0 / 2.5 A/B**：两版共用情感样本库和 QwenEmotion，映射表可以直接复用。在 5090 上用同一批角色、同一批句子盲听，同时比较音色稳定性和情感自然度，再决定主力版本。2.0 还有 TRT 后端可以提速。

---

## 8. 建议的最小验证清单（交给后续实现票）【推测】

1. Windows 原生执行 `uv sync --extra webui`，跑 `gpu_check.py`，再用 WebUI 合成一句，记录 bf16 和 fp32 下的 RTF（区分 7 字、30 字、80 字三档）。
2. 选 2 个角色，每人 10 句，覆盖 10 类情绪，按第 5 节的表在 2.5 和 2.0 上各合成一遍，计算说话人相似度并盲听。
3. “害羞/轻声/哭腔”分别用向量和同角色参考音频合成，对比效果。
4. 有余力再去 WSL2 里试 `use_accel`，同时听 FP16 有没有音质问题。

## 9. 未解决 / 未核实

- 公开的 `gpt.pth` 是 base 还是 RL 版，`s2mel.pth` 是 U-DiT 还是 Zipformer（#801 没有官方回复）。
- `triton-windows 3.1` 与 sm_120、flash-attn 在 Windows 上能否编译通过，都没有实测，也没找到一手报告。
- 5090 **笔记本**上 2.5 的实测 RTF 没有找到一手数据。第 4 节的区间是推测。
- DisCo-Speech 的 PDF 下载超时，表 3 的数字来自 arXiv HTML 版（v1）。v3 是否改动过这些数字没有核对。

---

## 来源（一手优先）

- 官方仓库 README（2.5 / 2.0 用法、RTF 表、安装）：https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/README.md
- 2.5 推理代码：https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/indextts/infer_v2_5.py
- 2.5 GPT 模型：https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/indextts/gpt/model_v2_5.py
- WebUI（向量归一化、精度选择、实验功能）：https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/webui.py
- 依赖与 extras：https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/pyproject.toml
- TRT 后端 README（同机 RTF 对比）：https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/backends/trt/README.md
- HF 模型卡与文件：https://huggingface.co/IndexTeam/IndexTTS-2.5 ，https://huggingface.co/IndexTeam/IndexTTS-2
- ModelScope：https://modelscope.cn/models/IndexTeam/IndexTTS-2.5
- IndexTTS 2.5 Technical Report：https://arxiv.org/abs/2601.03888
- IndexTTS2 论文：https://arxiv.org/abs/2506.21619
- DisCo-Speech：https://arxiv.org/abs/2512.13251 （HTML：https://arxiv.org/html/2512.13251v1）
- PR / issue：[#468](https://github.com/index-tts/index-tts/issues/468)、[#517](https://github.com/index-tts/index-tts/pull/517)、[#580](https://github.com/index-tts/index-tts/issues/580)、[#584](https://github.com/index-tts/index-tts/issues/584)、[#292](https://github.com/index-tts/index-tts/issues/292)、[#314](https://github.com/index-tts/index-tts/issues/314)、[#611](https://github.com/index-tts/index-tts/issues/611)、[#722](https://github.com/index-tts/index-tts/issues/722)、[#733](https://github.com/index-tts/index-tts/pull/733)、[#736](https://github.com/index-tts/index-tts/issues/736)、[#759](https://github.com/index-tts/index-tts/issues/759)、[#762](https://github.com/index-tts/index-tts/pull/762)、[#767](https://github.com/index-tts/index-tts/pull/767)、[#781](https://github.com/index-tts/index-tts/issues/781)、[#798](https://github.com/index-tts/index-tts/pull/798)、[#801](https://github.com/index-tts/index-tts/issues/801)
- PyTorch sm_120 支持（论坛维护者说明）：https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099
- RTX 5090 Laptop 规格：https://www.notebookcheck.com/Nvidia-enthuellt-GeForce-RTX-5090-5080-und-5070-Laptop-GPUs-mit-bis-zu-10-496-CUDA-Kernen-und-24-GB-GDDR7.943054.0.html
