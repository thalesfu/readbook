# 04: Qwen3-TTS 官方 PyTorch 版在 Windows 上推理

Type: research（AFK）
Status: resolved
Blocked by: 无（可以立即开始）

## Question

查清楚在 Windows 和 5090 上用官方 PyTorch 版 Qwen3-TTS-12Hz-1.7B-Base 做带文本复刻的事实：
- 安装方式；flash-attention 在 Windows 上的问题和绕过办法；
- 和 mlx-audio 版在参数上的差异：温度、`top_p`、参考音频长度要求、`x_vector_only` 模式；
- 有没有能调语速的参数（MLX 版的 `speed` 参数在复刻模式下没生效）；
- 批量推理和预期实时率。

产出调研笔记，标明事实和推测。

## Answer

调研笔记在分支 `research/qwen3-torch` 的 `research/qwen3-torch.md`（提交 2a054ef）。要点：

- **安装**：先装 torch cu128（Blackwell 从 2.7 起支持），再装 `qwen-tts`。不装 flash-attn，用默认的 sdpa；有人报告开 flash-attn 出现 NaN。dtype 用 bf16。
- **最大的迁移陷阱**：官方的复刻接口默认是流式布局，第三方实测参考音一长就越读越快；必须显式传 `non_streaming_mode=True`，这和 mlx 版的行为一致。
- **参数要对齐 mlx 版**：
  - 子 talker 的温度要另外设成 0.6（官方默认 0.9）；
  - mlx 在这个模式下把 repetition_penalty 抬到 1.5、只罚最近 64 个 token，官方是 1.05、罚全部历史，要对齐只能给官方代码打补丁（草稿见笔记 §9）。
- **语速**：官方没有语速参数，复刻用的 Base 版也不接受指令。只能靠非流式、参考音控制在 5–10 秒、切短句，再加后期变速。
- **批量和速度**：传文本列表就能批量。第三方数据：5090 上原版实时率约 1.6，Windows 更慢；社区 PR 用 batch=4 能到约 2.2 倍实时。"每句试 5 次"可以直接写成 batch=5，但要按句长设 `max_new_tokens`，防止一条失控拖住整批。
- **未实测**：以上速度数字和推荐参数都没在 5090 上验证过，笔记末尾附了验证清单，留给"在 5090 上复现混合版"那张票。
