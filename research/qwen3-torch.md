# Qwen3-TTS 官方 PyTorch 版在 Windows + RTX 5090 笔记本上做带文本复刻

调研日期：2026-10-26。对应票：`.scratch/voice-pipeline/issues/04-qwen3-torch-research.md`。

**读过的一手材料及版本**

- 官方仓库 [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)：main 分支 HEAD 是 [`022e286`](https://github.com/QwenLM/Qwen3-TTS/commit/022e286b98fbec7e1e916cb940cdf532cd9f488e)（2026-03-17，此后没有新提交）。
- PyPI 包 [`qwen-tts`](https://pypi.org/project/qwen-tts/)：最新 0.1.1（2026-02-06）。
- HF 模型卡与配置：[Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)。
- 本机装的 mlx-audio 0.6.1 源码：`mlx_audio/tts/models/qwen3_tts/qwen3_tts.py`。
- 官方仓库的 issue 和 PR。
- PyTorch、flash-attention、transformers 4.57.3 的官方源码和发布页。

**标注约定**：正文默认写的是**事实**，都给出来源（官方代码写到行号，下文简称 `modeling` = `qwen_tts/core/models/modeling_qwen3_tts.py`，`wrapper` = `qwen_tts/inference/qwen3_tts_model.py`，均按 commit 022e286）。**【推测】** 标出的是我根据这些事实推出来、还没有实测的判断。issue 里网友的测量数据算"第三方报告"，单独标出。

---

## 0. 结论速览

1. **安装**：用 `pip install qwen-tts`，它把 `transformers==4.57.3` 和 `accelerate==1.12.0` 钉死了。PyTorch 要自己先装 cu128 版：5090 是 Blackwell，至少要 2.7；cu128 这条线 Windows 上最新的是 2.11.0。**flash-attn 不用装**：官方没有 Windows 预编译轮子，而且 attn_implementation 不传时，transformers 4.57.3 默认就用 `sdpa`。
2. **头号迁移陷阱**：官方 `generate_voice_clone` 的 `non_streaming_mode` **默认是 False**（文本一边生成一边喂，即"模拟流式"）。mlx-audio 的 ICL 实现用的却是**非流式**的提示布局，等于官方的 `non_streaming_mode=True`。第三方实测（#239 / PR #362）显示：流式布局配上长参考音频，语速会越读越快；改成非流式，测得的漂移是 0。**在 Windows 上一定要显式传 `non_streaming_mode=True`**。
3. **采样参数不能照搬**：
   - 你在 Mac 上传了 `temperature=0.6`。mlx 版的子 talker（code predictor）**和主 talker 共用这个温度**；官方把两者分开，子 talker 默认 0.9。
   - mlx 版在 ICL 模式下会把 `repetition_penalty` **强行抬到 ≥1.5**，并且只罚最近 64 个 token；官方默认 1.05，罚的是**全部已生成的 token**。
   - 所以想复现 Mac 上的结果，至少要传 `subtalker_temperature=0.6`。重复惩罚这一项没法靠传参对齐（见 §4.3）。
4. **语速**：官方没有任何语速参数，Base 模型也不支持 instruct，复刻模式下能直接调语速的旋钮一个都没有。可用的办法：用非流式布局防漂移、参考音频控制在 5–10 秒左右、句子别太长，最后在后期做变速。
5. **批量和速度**：`generate_voice_clone` 原生支持把一批文本放进一个 list 一起跑（左填充）。原版 eager 路径很慢，原因是每生成一帧都要调一次 HF `generate` 去跑 15 个子码本。
   - 第三方数据：5090 台式机 Linux 原版单条 RTF（耗时÷音频时长）约 1.6；同一张卡在 Windows 上有人报到约 10 倍实时的慢速。
   - PR #380 在 5090 上测 batch=4 的吞吐约为实时的 2.2 倍，开 CUDA graph 后约 5 倍。
   - 每句"试 5 次取最好"正好可以写成 batch=5 一次跑完。

---

## 1. 安装

### 1.1 官方给的安装方式（事实）

- 官方 README 的 Quickstart：建一个干净的 Python 3.12 环境，`pip install -U qwen-tts`；想改源码就 `git clone` 后 `pip install -e .`。FlashAttention 2 写的是"推荐"而不是必须：`pip install -U flash-attn --no-build-isolation`，内存小于 96GB 时加 `MAX_JOBS=4`。（[README L104-142](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/README.md#L104-L142)）
- `qwen-tts` 0.1.1 声明的依赖（[pyproject.toml](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/pyproject.toml)，与 PyPI 元数据一致）是 `transformers==4.57.3`、`accelerate==1.12.0`、`gradio`、`librosa`、`torchaudio`、`soundfile`、`sox`、`onnxruntime`、`einops`。**依赖里没有 torch 本体**（是 torchaudio 把它间接拉进来的），所以 CUDA 版 torch 要你自己先装好。
- PyPI 上的版本历史：0.0.2–0.0.5 是 2026-01-22/23 发的，0.1.0 是 02-05，0.1.1 是 02-06。之后只有 2026-03-17 往仓库推过一次 finetuning 修复，没有再发 PyPI。
- 权重怎么下：`from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")` 第一次调用会自动从 HF 拉主模型和 `speech_tokenizer/` 子目录（`modeling` L1890-1936）。也可以先用 `huggingface-cli download` 或 ModelScope 下到本地目录再加载（README L84-102）。

### 1.2 Windows 特有的小坑（事实，后果属【推测】）

- 一 `import qwen_tts` 就会打印 `Warning: flash-attn is not installed...` 和 `SoX could not be found!`。原因是 `qwen_tts/core/__init__.py` 在加载包时就 import 了 25Hz tokenizer 的模块：flash-attn 的警告出自 `tokenizer_25hz/vq/whisper_encoder.py` L35，`import sox` 在 `tokenizer_25hz/vq/speech_vq.py` L16。**12Hz 模型的推理路径用不到 sox 和 onnxruntime**（grep 全包，只有 25Hz 的 `speech_vq.py` 引用了它们）。【推测】这两条警告在 12Hz Base 上可以不管。#2 里有人说装了 sox 以后"破音没了"，按代码看这多半是巧合。
- 参考音频从本地路径读入时走 `librosa.load(path, sr=None, mono=True)`（`wrapper` L207-223）。【推测】mp3 之类的格式在 Windows 上可能要额外装 ffmpeg。直接给 24k/48k 的 wav 最省事。

### 1.3 Blackwell（sm_120）和 CUDA 12.8（事实）

- PyTorch 从 2.7 开始支持 Blackwell，并提供 CUDA 12.8 预编译轮子（[PyTorch 2.7 发布博客](https://pytorch.org/blog/pytorch-2-7/)）。
- `download.pytorch.org/whl/cu128` 上 cp312 / win_amd64 的 torch 轮子有 2.7.0、2.7.1、2.8.0、2.9.0、2.9.1、2.10.0、2.11.0，torchaudio 同样到 2.11.0 为止。更新的 2.14.x 只出了 cu130/cu132 版（2026-10-26 查索引页得到）。
- 第三方旁证：
  - faster-qwen3-tts 的 README 写着"RTX 50xx / Blackwell GPUs need CUDA 12.8 PyTorch wheels ... (PyTorch 2.7+)"（[README](https://github.com/andimarafioti/faster-qwen3-tts)）。
  - #239 的环境是 RTX 5060 + CUDA 12.8，#350 用的是 `torch 2.9.1+cu128`，都能正常跑。
- 【推测】笔记本上写的"CUDA 12.8"如果指的是 `nvidia-smi` 右上角那个驱动能支持的最高 CUDA 版本，那就只能用 cu128 轮子，也就是 torch ≤2.11.0。如果驱动更新到能支持 CUDA 13，也可以用 2.14 的 cu130 轮子。`qwen-tts` 钉死的 transformers 4.57.3 对 torch 版本没有上限。

## 2. flash-attention 在 Windows 上的问题和绕过办法

### 事实

- flash-attention 官方 README 对 Windows 的原话是：Linux。"Might work for Windows starting v2.3.2 ... but Windows compilation still requires more testing"（[README](https://github.com/Dao-AILab/flash-attention#installation-and-features)）。
- 官方 FA2 支持的显卡列表只写了 Ampere、Ada、Hopper。不过当前 main 分支的 `setup.py` 默认编译目标是 `80;90;100;110;120`，CUDA ≥12.8 时会生成 sm_120 的代码（[setup.py](https://github.com/Dao-AILab/flash-attention/blob/main/setup.py)，`add_cuda_gencodes`）。
- PyPI 上最新的 `flash-attn` 2.8.3.post2（2026-10-14）**只有源码包**。GitHub Release 的 v2.8.3.post2 带了 48 个预编译轮子，**一个 win 的都没有**。安装时 setup.py 会先去下载匹配的预编译轮子，下不到就在本地编译。#41 里那个 Windows 用户遇到的 `HTTP Error 404` 加 `Python.h` 找不到，就是这条路径。
- **官方代码不要求 flash-attn**：
  - `Qwen3TTSModel.from_pretrained(**kwargs)` 把 `attn_implementation` 原样传给 HF（`wrapper` L83-121）。
  - 模型类声明了 `_supports_sdpa = True`（`modeling` L473-474）。
  - transformers 4.57.3 在 `attn_implementation` 为 None 时默认选 `"sdpa"`（[modeling_utils.py L2695-2725 @v4.57.3](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/modeling_utils.py#L2695)）。
  - 官方 demo 也有 `--no-flash-attn` 开关，关掉后传的是 `attn_implementation=None`（`qwen_tts/cli/demo.py` L103-109、L606）。维护者在 [#12](https://github.com/QwenLM/Qwen3-TTS/issues/12) 里也是让大家用 `--no-flash-attn`。
- 和 FA2 有关的已知问题：
  - [#333](https://github.com/QwenLM/Qwen3-TTS/issues/333)：在 A100 上用 FA2 跑声音复刻评测会出现 NaN logits，触发 `device-side assert`；换成 `sdpa` 或 `eager` 就正常。
  - [#136](https://github.com/QwenLM/Qwen3-TTS/issues/136)：开 FA2 时总报 "without specifying a torch dtype" 的警告（PR #309 想修，还没合并）。
- 速度上，第三方报告（[#89](https://github.com/QwenLM/Qwen3-TTS/issues/89)）：
  - 在 5070 Ti 和 5090 上，不开 flash-attn 反而快大约 15%。
  - 做了 torch.compile 优化的 fork 里，FA2 和 eager 速度没有可见差别。
  - 有人在 Windows 上为 50 系编译出了 FA2，但同一个 issue 里也有人反映"装了 FA2，RTF 还是 ~2"。

### 绕过办法

直接用 `attn_implementation="sdpa"`，或者干脆不传（默认就是 sdpa）。dtype 用 `torch.bfloat16`。【推测】这个模型 decode 阶段每步只处理一个 token，单步的 attention 很小，瓶颈在 Python 和 kernel 启动开销（见 §6），所以 FA2 对速度帮助有限，不值得在 Windows 上折腾编译。

另外别用 fp16：[#43](https://github.com/QwenLM/Qwen3-TTS/issues/43) 报告 fp16 在 Turing 卡上会出 NaN、触发 assert。5090 原生支持 bf16，没必要用 fp16。

## 3. 带文本复刻的调用方式（官方 API，事实）

- 入口是 `Qwen3TTSModel.generate_voice_clone(text, language, ref_audio, ref_text, x_vector_only_mode=False, voice_clone_prompt=None, non_streaming_mode=False, **kwargs)`（`wrapper` L470-635）。
- 同一个参考音频反复使用时，可以先用 `create_voice_clone_prompt(ref_audio, ref_text, x_vector_only_mode)` 建一次 prompt 缓存起来，之后通过 `voice_clone_prompt=` 传入，避免每次重新编码参考音频（README L266-285；`wrapper` L356-458）。
- 这个 prompt 里有两样东西：
  - `ref_code`：12Hz tokenizer 编出来的 (T, 16) 码。
  - `ref_spk_embedding`：把参考音频重采样到 24k，算 mel 谱，再过 speaker encoder 得到的 x-vector。
- ICL 模式下 **x-vector 和参考码都会放进提示里**（`modeling` L2102-2106：`x_vector_only_mode or icl_mode` 时都会注入 speaker_embed）。mlx 版也一样（`qwen3_tts.py` L767-770，注释写的是 "ICL still uses x-vector"）。
- 输出时，先把参考码拼在生成码前面一起 decode，再**按码长比例**切掉前面属于参考的那段波形（`wrapper` L612-631）。mlx 版的做法完全相同（`qwen3_tts.py` L2495-2520）。

## 4. 与 mlx-audio 版的参数差异（核心）

### 4.1 对照表

"mlx 实际生效值"指你在 Mac 上的调用 `model.generate(text, ref_audio, ref_text, lang_code="chinese", temperature=0.6)`（见主仓库 `readbook/passage.py` L75-76、`bakeoff.py` L392），按 mlx-audio 0.6.1 的代码推出来的实际值。

| 参数 | mlx-audio 0.6.1（ICL） | 官方 PyTorch | 要对齐的话 |
|---|---|---|---|
| `temperature` | 你传的 0.6 | 默认 0.9（generation_config.json） | 传 `temperature=0.6` |
| 子 talker 温度 | **和主 talker 共用**，也是 0.6（`qwen3_tts.py` L2358-2363，子码本采样用的是同一个 `temperature/top_k/top_p`） | 单独的 `subtalker_temperature`，默认 0.9 | **传 `subtalker_temperature=0.6`** |
| `top_k` / `subtalker_top_k` | 50 / 共用 50 | 50 / 50 | 不用改 |
| `top_p` / `subtalker_top_p` | 1.0 / 共用 1.0 | 1.0 / 1.0 | 不用改 |
| `subtalker_dosample` | 没有这个开关，始终采样 | 默认 True | 不用改 |
| `repetition_penalty` | ICL 下会被强行改成 `max(rp, 1.5)` = **1.5**（L1276-1288，注释说是"防止长参考预填时码退化"），**只罚最近 64 个 token**（`_sample_token` 的 `repetition_context_size=64`，L837、L855-871） | 默认 1.05，用 HF `RepetitionPenaltyLogitsProcessor`。因为传进去的是 `inputs_embeds`，HF 的 `input_ids` 从空张量 `(B,0)` 开始（[generation/utils.py @v4.57.3](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/generation/utils.py)，`_maybe_initialize_input_ids_for_generation`），所以罚的是**全部已生成的首码本 token**，不限窗口 | 传参只能改系数，改不了窗口。见 §4.3 |
| 被罚的对象 | 首码本（主 talker）。子 talker 不罚 | 首码本。子 talker 的 `code_predictor.generate` 不传 rp（`modeling` L1671-1680） | 一致 |
| `max_new_tokens` | 取 generation_config 的 8192 | 也取 generation_config.json 的 8192（hard default 2048 只在缺这个文件时才用） | 建议按句长设上限，见 §6.3 |
| `min_new_tokens` | 没有 | 写死为 2（`modeling` L2046） | 差异可以忽略 |
| 被压制的 token | 词表最后 1024 个里除 EOS 以外全部压制 | 同样（`modeling` L2059-2063） | 一致 |
| 语言 | `lang_code="chinese"` 映射到 language_id 2055，走 think 前缀 | `language="Chinese"` 也一样（`modeling` L2110-2148） | 传 `language="Chinese"` |
| **文本布局** | 非流式：整段参考文本加目标文本先压在 codec PAD 上预填，再接参考码（`_prepare_icl_generation_inputs` L746-760，`trailing_text_hidden = tts_pad_embed`） | **`generate_voice_clone` 默认 `non_streaming_mode=False`**（`wrapper` L478）。CustomVoice/VoiceDesign 在 [0c6a7cb](https://github.com/QwenLM/Qwen3-TTS/commit/0c6a7cbb6c8421a46332f8c2434c7825c4c855ef) 里已经改成默认 True，clone 的默认值没跟着改 | **传 `non_streaming_mode=True`** |
| `speed` | 有这个参数，docstring 写着 "not directly supported yet"，实际不会传给任何生成逻辑（只在 `supports_tts_batch` 里用来判断能不能批量，L230） | 根本没有这个参数 | 见 §5 |
| x_vector_only | 没有显式参数。只传 `ref_audio` 不传 `ref_text` 时，会退化到只用 speaker embedding 的路径（L410-411、L1270-1275） | `x_vector_only_mode=True`，此时 `ref_text` 可以不填，官方说"cloning quality may be reduced"（README L243） | 我们要的是带文本复刻，保持 False |
| 随机种子 | 没有参数 | 没有参数。用 `torch.manual_seed` / `transformers.set_seed` 控制全局随机（#333 的脚本就是这么做的） | — |

两边**一致**的部分（事实）：
- ICL 提示的拼法：role 前缀、think/语言/speaker/pad/bos 前缀、`ref_id[:,3:-2]` 和 `text_id[:,3:-5]` 的切片方式。mlx 的 docstring 明说这部分"Matches the official Qwen3-TTS generate_icl_prompt structure"，我逐段对过 `modeling` L1968-2020 和 L2126-2200，在非流式分支下是同构的。
- 解码方式和切掉参考段的比例算法。

### 4.2 `non_streaming_mode` 为什么要紧（事实＋第三方测量）

代码层面（`modeling` L1968-2020，`generate_icl_prompt`）：

- `non_streaming_mode=True` 时：全部文本（参考文本＋目标文本＋eos）叠在 codec PAD 上一次性预填，然后接参考码叠 tts_pad。之后每生成一帧，文本侧都只喂 pad。
- `non_streaming_mode=False` 时：文本嵌入和参考码嵌入**逐位置相加**。文本比参考码短的话，剩下的位置补 pad。文本更长的话，多出来的部分变成 `trailing_text_hidden`，生成时**每帧喂一个 token**（喂法在 `modeling` L1690-1693）。

第三方实测（[#239](https://github.com/QwenLM/Qwen3-TTS/issues/239) 的 2026-08-28 评论和 [PR #362](https://github.com/QwenLM/Qwen3-TTS/pull/362)，作者在 MPS 上固定种子测"音节核/秒"，比较每次生成的头 1/3 和尾 1/3）：

| 配置 | 一次约 25 秒的生成内，语速变化 |
|---|---|
| 流式，参考 12.9 秒 | +16.7% |
| 流式，同一段参考只截 4 秒 | −2.3% |
| 非流式，参考 12.9 秒 | 0.0% |

- 他们给出的机制：流式模式下文本按每帧一个 token 的速度喂进去，也就是 12.5 token/秒，比说话消耗文本的速度快得多。生成前期模型说不到还没收到的字，这相当于一个隐性的"刹车"。参考越长，预填阶段就已经喂掉越多目标文本，刹车也就越早失效。
- 在流式模式下，五段语速从 112 到 182 wpm 不等的参考，输出都收敛到约 195 wpm，**和参考本身的语速无关**。这和 [#290](https://github.com/QwenLM/Qwen3-TTS/issues/290) 的现象对得上："把参考放慢 20%，输出仍然被压缩到 41–48 秒"。
- 他们还试过几种在推理侧修复流式喂法的方案，全部失败（立即 EOS、生成失控、语速忽快忽慢），所以建议的绕法就是用非流式。PR #362 截至调研时还**没有合并**（状态 OPEN）。

【推测】Mac 上 mlx 版能读得稳，很可能部分得益于它本来就是非流式布局。Windows 上如果照官方默认跑，就会碰上"语速变快"。这一条是迁移时最可能造成"效果复现不了"的原因。

### 4.3 重复惩罚怎么对齐

- 事实：`Qwen3TTSForConditionalGeneration.generate` 只把它自己列出的参数组装成 `talker_kwargs` 传给 `self.talker.generate(...)`。`**kwargs` **不会往下传**（`modeling` L2022-2066、L2272-2278）。README 里"可以传任何 HF generate 参数"的说法在这条路径上不成立，所以通过 `generate_voice_clone(..., logits_processor=...)` 塞自定义处理器是行不通的。
- 事实：`self.talker` 是一个普通的 HF `GenerationMixin`，没有重写 `generate`（`modeling` L1564）。在 transformers 4.57.3 里，自定义的 `logits_processor` 会排在内置处理器之后、temperature/top-k/top-p 这些 warper 之前（`generation/utils.py` L1248-1267）。
- 【推测】想严格复现"1.5 只罚最近 64 个"，可以包一层 `tts.model.talker.generate`：把 `repetition_penalty` 设成 1.0，再塞一个只看最近 64 个 token 的窗口惩罚 LogitsProcessor（代码草稿见 §9）。另一个思路是先试官方原生的 1.05（全历史），听感能接受就不改。**两种都没实测过**。全历史 1.5 对长句的影响可能比 1.5@64 窗口更猛，不建议直接传 `repetition_penalty=1.5`。

### 4.4 参考音频长度（事实＋第三方）

- 官方只说了下限：模型卡写的是"3-second rapid voice clone"（HF 模型卡模型表）。没有给上限，代码里也没有截断或长度检查。参考码会整段进入上下文（`max_position_embeddings=32768`，见 config.json）。
- 参考太长有风险：流式布局下会让语速变快（§4.2，PR #362 建议流式下参考 <5 秒）。参考的长短也直接决定预填长度。
- 参考太短有风险：[#68](https://github.com/QwenLM/Qwen3-TTS/issues/68) 里用"嗯？"这种极短的参考，会出现生成停不下来的情况。
- 参考的尾巴会被"读出来"（[#341](https://github.com/QwenLM/Qwen3-TTS/issues/341)，OPEN）：ICL 生成的开头会复读 `ref_text` 的最后几个词。调 rp、改成 greedy 都没用。有人把参考尾部那个短语删掉后就好了。faster-qwen3-tts 的 README 把这归因于"参考在词中间截断，导致首帧受最后那个音素影响"，它默认在参考末尾**补 0.5 秒静音**。
- 偶尔会换成别人的声音（[#80](https://github.com/QwenLM/Qwen3-TTS/issues/80)）：有人用 4–20 秒的英文参考，约 20% 的请求全部或部分不像参考；重新生成一次能解决。参考语言和目标语言不一致时更容易出现。引号开头的直接引语也容易换声音（2026-09 的评论）。

【推测】起点建议参考 5–10 秒，干净，结尾落在完整的一句话上并带一小段静音，`ref_text` 要逐字准确。你们现有的锚点参考如果已经满足这些，就不用动。

## 5. 能不能控制语速

- 事实：
  - 官方 `qwen_tts` 全包里没有任何 speed、rate、duration 之类的生成参数（grep 确认）。
  - `generate_voice_clone` 没有 `instruct` 参数。PR #281 想给 Base 预留 instruct，标题写的是"for upcoming 25Hz model"，没有合并。
  - mlx-audio 的 `speed` 在复刻模式下完全不起作用（§4.1）。
- 事实（第三方）：在官方默认的流式布局下，参考音频的语速不会被继承（#290，PR #362）。faster-qwen3-tts 给 Base 加了实验性的 `instruct`，它自己也说 ICL 下"behaved much more predictably"，但这已经不是官方能力。
- 【推测】可用的手段按推荐顺序：
  1. 用 `non_streaming_mode=True` 消除越读越快的漂移，这是防加速最关键的一步；
  2. 选语速合适的参考（非流式下参考语速能继承多少，没有测量数据，需要自己 A/B）；
  3. 句子切短，issue 里建议单次生成不超过约 30 秒音频；
  4. 后期变速，比如保持音高的 time-stretch。你们流水线里已经有"语速拉回目标的 0.85–1.2 倍"这一步，可以沿用；
  5. 继续用"每句试 5 次，按语速和相似度挑最好的"。

## 6. 批量推理和实时率

### 6.1 批量接口（事实）

- `generate_voice_clone(text=[...], language=[...] 或单个字符串, voice_clone_prompt=prompt_items)`：prompt 只有 1 个而文本有多条时，会自动复制给每一条（`wrapper` L566-588）。官方示例 `examples/test_model_12hz_base.py` 演示了"一个 prompt 配多条文本"和"多个 prompt 配多条文本"。
- 批内各条的输入嵌入做**左填充**，并配上 attention mask。流式布局下剩余的文本隐状态按 pad 对齐（`modeling` L2239-2269）。生成结束后，每一条按各自第一个 EOS 截断（L2280-2292）。
- 多线程并发调用同一个实例会比串行还慢。[#350](https://github.com/QwenLM/Qwen3-TTS/issues/350) 说得很清楚：要并发就用 batch list、队列加单 worker，或者开多个实例。

### 6.2 速度：为什么慢，以及有哪些数字

- 事实（代码）：主 talker 每生成一帧，都要在 `forward` 里调一次 HF `code_predictor.generate(max_new_tokens=15)` 来补齐剩下 15 个码本（`modeling` L1669-1680）。也就是说，每帧要跑一次完整的 HF generate 循环，Python 和 kernel 启动开销很大。
- 第三方数据。下面 RTF 的口径不统一：issue #89 里的 RTF 是"耗时÷音频时长"，越小越快；faster-qwen3-tts 和 PR #380 用的是"音频时长÷耗时"，越大越快。

| 来源 | 硬件 / 系统 | 配置 | 结果 |
|---|---|---|---|
| [#89](https://github.com/QwenLM/Qwen3-TTS/issues/89) 楼主 | 5090，Windows 11，CUDA 13.1，torch 2.9/2.10 | 原版 | 约 10 倍于实时（慢），GPU 利用率 4–5% |
| #89 RyrieNorth | 5090，Linux，cu130，torch 2.10，FA 2.8.3 | 原版非流式 | 耗时÷音频 = **1.57** |
| 同上 | 同上 | dffdeeq fork（torch.compile 等） | 0.50 |
| #89 dffdeeq | 5090 | 自己的 fork 优化后 | <0.25（后来又宣称再快约 2 倍） |
| #89 dur-randir | 同一台机器 Windows→Linux | 原版 / fork | Linux 快约 3 倍（他怀疑是 CPU 争用） |
| #89 nfranke | 5070 Ti，CUDA 12.9，不开 FA | 原版 / fork | 0.75 / 0.27 |
| [PR #380](https://github.com/QwenLM/Qwen3-TTS/pull/380)（OPEN） | 5090，torch 2.14.1+cu132，SDPA，bf16，1.7B Base ICL | eager batch=4 / CUDA graph batch=4 | 音频÷耗时 2.18× / 5.07×，显存峰值 5.2 / 9.2 GiB |
| 同上 | 同上 | eager batch=8 / CUDA graph batch=8 | 4.04× / 7.28×，显存峰值 6.5 / 14.5 GiB |
| faster-qwen3-tts README | RTX 4090 / RTX 4060（Windows），1.7B | 原版 / CUDA graph | 4090：0.82× → 4.22×；4060 Win：0.23× → 1.83×（音频÷耗时） |

### 6.3 【推测】对 5090 笔记本 + Windows 的预期

- 笔记本版 5090 的算力和显存带宽都比台式机 5090 低一档，这是我的背景知识，这次没有核实。再加上 Windows 上 Python/CUDA 调度开销更大（#89 的两份报告），原版 eager 单条推理的 RTF（耗时÷音频）大概会落在 **1.5–4** 之间，也就是比实时慢。
- 用 batch 能明显提高吞吐：每句 5 次尝试正好凑成 batch=5，预计和单条耗时差不多，就能拿到 5 个候选。按 PR #380 batch=4 约 2.2 倍于实时、笔记本打个折来估，5 个候选的总耗时约等于这句音频时长的 0.5–1.5 倍。
- 批内最慢的一条决定整批耗时。生成失控（#118：约 0.5% 的概率不出 EOS，一直跑到 max_new_tokens）会把整批拖死，所以**一定要按句长设 `max_new_tokens`**。12Hz 模型每秒 12.5 帧，中文大约 4–5 字/秒，建议上限取 `字数 × 6 + 60` 帧左右（≈ 字数 × 0.5 秒 + 5 秒）。
- 24GB 显存对 1.7B bf16 加 batch=5–8 很宽裕（PR #380 的峰值显存远低于 24GB）。
- 想进一步提速，可以看 faster-qwen3-tts（CUDA graph，有 `setup_windows.bat`）或者 PR #380 的 `use_cuda_graph`。代价是偏离官方代码路径，而且它们都声明过"结果不是逐位一致"。vLLM-Omni 官方 README 写的是只支持离线推理，没查到 Windows 原生支持的说法，不推荐用。

## 7. 官方 issue 中和"音色漂移、语速变快、Windows"有关的已知问题

| issue | 状态 | 要点 |
|---|---|---|
| [#239](https://github.com/QwenLM/Qwen3-TTS/issues/239) 长文本语速逐渐变快 | closed（被自动关闭） | 1.7B Base 复刻时，超过约 100 字的文本越往后越快。机制见 §4.2，绕法是 `non_streaming_mode=True` |
| [PR #362](https://github.com/QwenLM/Qwen3-TTS/pull/362) | open | 提议把 clone 的默认值改成 `non_streaming_mode=True`，附测量数据 |
| [#290](https://github.com/QwenLM/Qwen3-TTS/issues/290) 复刻无视参考语速，总是更快 | closed（被自动关闭），2026-10 仍有人回复"same issue" | temperature、subtalker、do_sample 调来调去都没用 |
| [#179](https://github.com/QwenLM/Qwen3-TTS/issues/179) 微调 Base 后每个 epoch 语速都更快 | open | 只和微调有关，零样本复刻不受影响 |
| [#80](https://github.com/QwenLM/Qwen3-TTS/issues/80) 复刻偶尔换人 | closed | 约 20% 的请求部分或全部不像参考；重新生成能解决；跨语言、直接引语时更容易出现 |
| [#61](https://github.com/QwenLM/Qwen3-TTS/issues/61) 多人对话里突然变成别的声音 | closed | 中英混用时 100% 复现，纯中文时没问题（Windows 11，5060 Ti） |
| [#341](https://github.com/QwenLM/Qwen3-TTS/issues/341) ICL 开头复读参考尾部 | open | 见 §4.4 |
| [#68](https://github.com/QwenLM/Qwen3-TTS/issues/68) 复刻的若干坑 | closed | 短文本生成停不下来要手动设 max_new_tokens；结尾吞字；连续生成约 4 分钟后开始乱 |
| [#181](https://github.com/QwenLM/Qwen3-TTS/issues/181) 长文本最后几个词被吞 | closed | 10 秒参考、300 词时容易出现 |
| [#118](https://github.com/QwenLM/Qwen3-TTS/issues/118) 偶尔不出 EOS | closed | 约 1/200 的概率一直跑到 max_new_tokens |
| [#89](https://github.com/QwenLM/Qwen3-TTS/issues/89) 5090 推理很慢 | closed | 楼主就在 Windows 上；Linux 快约 3 倍；fork 优化方案见 §6.2 |
| [#132](https://github.com/QwenLM/Qwen3-TTS/issues/132) Blackwell 上效率低 | closed | RTX 5050 的 CUDA 利用率不到 15%，楼主最后转用 dffdeeq 的 fork |
| [#41](https://github.com/QwenLM/Qwen3-TTS/issues/41)、[#12](https://github.com/QwenLM/Qwen3-TTS/issues/12)、[#293](https://github.com/QwenLM/Qwen3-TTS/issues/293) flash-attn 装不上或者报警告 | closed | Windows 和新 Ubuntu 上都编译失败；不装 FA、改用 `--no-flash-attn` 或 sdpa 就行 |
| [#333](https://github.com/QwenLM/Qwen3-TTS/issues/333) 开 FA2 出 NaN logits | closed | 换 sdpa 或 eager 就正常 |
| [#298](https://github.com/QwenLM/Qwen3-TTS/issues/298) 同样输入每次输出不同 | closed | 只能固定种子；就算固定了种子，结果也不是逐位一致 |

## 8. 推测汇总（待在 5090 上实测验证）

1. 不传 `non_streaming_mode=True` 的话，Windows 版会比 Mac 版读得更快，长句尤其明显。
2. 只对齐 `temperature` 不对齐 `subtalker_temperature`，音色和音质的随机性会比 Mac 上大（子 talker 0.9 对 0.6）。
3. 1.5@64 窗口和官方 1.05 全历史这两种重复惩罚的听感差异没测过，需要 A/B。
4. 原版 eager 在 Windows 笔记本上的单条 RTF 估计在 1.5–4（慢于实时），batch=5 时每句的"5 次尝试"大约花 1 倍音频时长。
5. flash-attn 对这个模型的提速有限，不值得在 Windows 上编译。
6. bf16 下 PyTorch 和 MLX 的数值不会逐位一致，同样的参数也只能做到"统计上等价"，不可能逐样本复现。

## 9. Windows 上的推荐安装与参数起点

### 安装（PowerShell，Python 3.12）

```powershell
# 1) 先看驱动能支持的最高 CUDA 版本（nvidia-smi 右上角）。12.8 就用 cu128。
py -3.12 -m venv .venv-qwen3; .\.venv-qwen3\Scripts\Activate.ps1
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
# 2) 官方包（会钉死 transformers==4.57.3、accelerate==1.12.0）
pip install qwen-tts==0.1.1
# 3) 不装 flash-attn。验证一下：
python -c "import torch;print(torch.__version__, torch.cuda.get_device_capability(0))"   # 期望 (12, 0)
```

【推测】如果 `pip install qwen-tts` 想把 torch 换成 CPU 版或者别的 CUDA 版，就先装 qwen-tts，再用上面的 index-url 加 `--force-reinstall` 把 torch/torchaudio 装回 cu128 版。

### 参数起点（对齐 Mac 现状）

```python
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa",
)
prompt = tts.create_voice_clone_prompt(ref_audio="anchor.wav", ref_text="……参考逐字稿……")  # 每个锚点建一次，缓存起来

def synth_candidates(text: str, n: int = 5):
    wavs, sr = tts.generate_voice_clone(
        text=[text] * n, language="Chinese", voice_clone_prompt=prompt,
        non_streaming_mode=True,          # 关键：和 mlx 的布局一致，防止越读越快
        temperature=0.6, subtalker_temperature=0.6,   # mlx 的子 talker 也是 0.6
        top_k=50, subtalker_top_k=50, top_p=1.0, subtalker_top_p=1.0,
        repetition_penalty=1.05,          # 官方默认。要复现 mlx 的 1.5@64，见下面的补丁
        max_new_tokens=len(text) * 6 + 60,  # 防止失控把整批拖死
    )
    return wavs, sr   # 然后沿用现有 QC：CAM++ 相似度、变调、语速，挑最好的
```

【推测，未测】想把重复惩罚对齐成 mlx 的 1.5@64 窗口：

```python
from transformers import LogitsProcessor, LogitsProcessorList

class WindowedRepPenalty(LogitsProcessor):
    def __init__(self, penalty=1.5, window=64): self.p, self.w = penalty, window
    def __call__(self, input_ids, scores):
        if input_ids.shape[1] == 0: return scores
        ids = input_ids[:, -self.w:]
        s = scores.gather(1, ids)
        s = torch.where(s < 0, s * self.p, s / self.p)
        return scores.scatter(1, ids, s)

_orig = tts.model.talker.generate
def _patched(*a, **kw):
    kw["repetition_penalty"] = 1.0
    kw["logits_processor"] = LogitsProcessorList([WindowedRepPenalty(1.5, 64)])
    return _orig(*a, **kw)
tts.model.talker.generate = _patched
```

### 验证清单（建议在 5090 上先跑一遍）

1. 拿同一组句子，分别跑 `non_streaming_mode=True` 和 `False`，用 `qc` 里现有的语速指标比较前 1/3 和后 1/3，确认 §4.2 的结论在中文上也成立。
2. `subtalker_temperature` 取 0.6 和 0.9 各跑一遍，比较 CAM++ 相似度的分布。
3. 重复惩罚三档 A/B：1.05 全历史、1.5@64 补丁、1.5 全历史。
4. 记下 batch=1 和 batch=5 的 wall time 和显存，填掉 §6.3 里那些推测的数字。
