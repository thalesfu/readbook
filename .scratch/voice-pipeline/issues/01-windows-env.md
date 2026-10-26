# 01: 在 5090 上搭好可复现的基础环境

Type: task（HITL）
Status: open
Blocked by: 无（可以立即开始）

## Question

在 Windows 的 RTX 5090 笔记本上，把后续所有原型要用的基础环境搭好并验证：
- CUDA 12.8、PyTorch cu128、ffmpeg、uv、Claude Code；
- readbook 仓库能跑；
- 能读到游戏语音包 `VOICE.PAK` 和剧本（中日文），也就是 gameresource 仓库里的 `lscr`、`lacpak`、`daily` 能用；
- 代码里写死的 Mac 路径（游戏目录、gameresource 仓库、IndexTTS2 环境、模型缓存）改成可配置。

**需要人操作的部分**（列成清单交给用户）：系统层面的安装；游戏文件放到哪里；是否开 SSH。

**完成标准**：在 Windows 上，`readbook.wa2.load_day` 能取出〈2002〉的 65 句，并能导出其中任意一句的原声 wav。
