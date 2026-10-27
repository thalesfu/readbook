# 13: 开通火山引擎语音合成

Type: task（HITL）
Status: resolved
Blocked by: 无（可以立即开始）

## Question

用户在火山引擎控制台开通豆包语音合成 2.0（以及音色设计），把 AppID 和 Access Token 设成环境变量，告诉 agent 变量名。agent 不代填、不经手凭证。

**完成标准**：agent 能用这组凭证合成一句测试语音。

## Answer

完成标准已经达到：

```
uv run python -m readbook.volc "这是豆包语音合成二点零的测试。窗外的雪下得很安静，像是谁也不想吵醒。"
```

- **结果**：合成出 6.0 秒的 MP3（24kHz 单声道），请求耗时 1.8 秒，计费 34 字。音色用的是 Vivi 2.0（`zh_female_vv_uranus_bigtts`）。
- **输出位置**：`out/volc/`。

**开通情况**（2026-10-27，用户确认后由 agent 在控制台操作）：
- 在豆包语音新版控制台（default 项目）开通了**语音合成 2.0**，同意了《AI中台服务专用条款》《退费说明》《生成式模型专用条款》。
- 计费：先用 20,000 字免费额度，用完自动按 0.0003 元/字后付费，平台不另行提醒。按这个价，全书 168 万字大约 500 元。
- **音色设计没开通**：它用的是单独计费的"音色槽位"，有预付费和后付费两种。14 号票要用的时候，再看价格决定开不开。

**凭证**：
- 新版控制台只用一个 API Key 鉴权（请求头 `X-Api-Key`），不再是 AppID 加 Access Token。资源 ID `seed-tts-2.0` 决定了用哪一代模型，也决定计费方式；这个资源 ID 只能配 2.0 的音色。
- 在控制台建了一个叫 readbook 的 API Key。
- 密钥本身由用户复制，设成用户环境变量 `VOLC_TTS_API_KEY`，记在了 vault 的技术账号笔记里。agent 没有经手密钥的值。
- `readbook/volc.py` 先从进程环境读这个变量，没有的话再去注册表 `HKCU\Environment` 里找。所以变量设好后，已经开着的会话不重启也能用。

**做了什么**：新增 `readbook/volc.py`，接的是 HTTP Chunked 单向流式 V3 接口（`https://openspeech.bytedance.com/api/v3/tts/unidirectional`）：
- 只发文字，不上传任何原声；
- 逐行读 JSON，把 base64 音频拼起来；收到 `code=20000000` 就结束，并带回计费字数；
- 只用标准库，不加依赖。

**留给 14 号票**：
- 2.0 的音色可以用 `audio_params.emotion`（只有部分音色支持）、`speech_rate`、`loudness_rate` 调；`req_params.model` 可以选 `seed-tts-2.0-expressive`（表现力更强，支持语音指令）。文档只把这个参数写在"声音复刻 2.0"名下，用在普通 2.0 音色上管不管用要实测。
- `enable_subtitle` 能拿到字级时间戳（只对 2.0 有效）。
- 音色列表：https://www.volcengine.com/docs/6561/1257544
- **硬约束仍然有效**：声音复刻要上传参考音频，**不能用声优原声去复刻**。
