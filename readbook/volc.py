"""火山引擎豆包语音合成 2.0（新版控制台，HTTP Chunked 单向流式 V3）。

凭证只从环境变量 VOLC_TTS_API_KEY 读，代码里不写、不打印。Windows 上新设的用户环境变量，
已经开着的进程看不到，所以环境里没有时再去注册表 HKCU\\Environment 里找。
只发文字，不上传任何原声。

每次请求的计费字数记在 out/volc/usage.jsonl（时间、音色、用途、计费字数、耗时），`用量` 汇总。
免费额度 20,000 字，用完自动按 0.0003 元/字后付费，平台不提醒，所以每次都要记。

    uv run python -m readbook.volc "要合成的一句话" [音色]   # 写到 out/volc/
    uv run python -m readbook.volc 用量                      # 汇总计费字数

可调的（14 号票实测哪些管用）：
- emotion / emotion_scale：只有部分音色支持，见音色列表的"多情感音色"；
- speech_rate / loudness_rate：-50～100，0 是原样，100 是 2 倍；
- context：语音指令（additions.context_texts，只对 2.0 音色有效，只取第一句，不计费）；
- model：seed-tts-2.0-expressive（文档写在声音复刻 2.0 名下）；
- section_id：多轮上下文，同一个 ID 的请求会参考之前的合成。
"""

import base64
import json
import os
import sys
import time
import urllib.request
import uuid

from .config import ROOT

URL = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
RESOURCE = "seed-tts-2.0"                # 只能用"豆包语音合成模型2.0"的音色，按字数计费
SPEAKER = "zh_female_vv_uranus_bigtts"   # Vivi 2.0，测试用
KEY_VAR = "VOLC_TTS_API_KEY"
OUT = ROOT / "out" / "volc"
LEDGER = OUT / "usage.jsonl"
FREE, PRICE = 20_000, 0.0003             # 免费字数；之后每字多少元


def api_key() -> str:
    key = os.environ.get(KEY_VAR)
    if not key and sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                key = winreg.QueryValueEx(k, KEY_VAR)[0]
        except FileNotFoundError:
            pass
    if not key:
        raise SystemExit(f"没有找到环境变量 {KEY_VAR}")
    return key


def synth(text: str, speaker: str = SPEAKER, fmt: str = "mp3", sample_rate: int = 24000, *,
          emotion: str | None = None, emotion_scale: int | None = None, speech_rate: int = 0,
          loudness_rate: int = 0, context: str | None = None, model: str | None = None,
          section_id: str | None = None, purpose: str = "") -> tuple[bytes, dict]:
    """合成一段文字，返回 (音频字节, 结束包)。结束包里有 usage（计费字数）和 secs（请求耗时）。
    每次请求都记进 LEDGER，purpose 写这次是干什么的。"""
    audio_params = {"format": fmt, "sample_rate": sample_rate}
    if emotion:
        audio_params["emotion"] = emotion
        if emotion_scale:
            audio_params["emotion_scale"] = emotion_scale
    if speech_rate:
        audio_params["speech_rate"] = speech_rate
    if loudness_rate:
        audio_params["loudness_rate"] = loudness_rate
    additions = {}
    if context:
        additions["context_texts"] = [context]
    if section_id:
        additions["section_id"] = section_id
    req_params = {"text": text, "speaker": speaker, "audio_params": audio_params}
    if model:
        req_params["model"] = model
    if additions:
        req_params["additions"] = json.dumps(additions, ensure_ascii=False)
    body = {"user": {"uid": "readbook"}, "req_params": req_params}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
        "Content-Type": "application/json",
        "X-Api-Key": api_key(),
        "X-Api-Resource-Id": RESOURCE,
        "X-Api-Request-Id": str(uuid.uuid4()),
        "X-Control-Require-Usage-Tokens-Return": "text_words",
    })
    t0 = time.perf_counter()
    audio = bytearray()
    end = None
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            logid = resp.headers.get("X-Tt-Logid")
            for line in resp:
                if not line.strip():
                    continue
                msg = json.loads(line)
                if msg.get("code") == 20000000:
                    end = {**msg, "logid": logid, "secs": round(time.perf_counter() - t0, 2)}
                    break
                if msg.get("code") not in (0, None):
                    raise RuntimeError(f"合成失败：{msg.get('code')} {msg.get('message')}（logid {logid}）")
                if msg.get("data"):
                    audio += base64.b64decode(msg["data"])
    finally:
        record(speaker, purpose, text, end)
    if end is None:
        raise RuntimeError(f"没有收到结束包（logid {logid}）")
    return bytes(audio), end


def record(speaker: str, purpose: str, text: str, end: dict | None):
    """记一笔：失败的请求也记（计费字数记 None），免得漏算。"""
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    words = (end or {}).get("usage", {}).get("text_words")
    with LEDGER.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "speaker": speaker, "purpose": purpose,
                            "chars": len(text), "billed": words, "secs": (end or {}).get("secs")},
                           ensure_ascii=False) + "\n")


def usage() -> dict:
    """按用途汇总计费字数。"""
    total, by = 0, {}
    if LEDGER.exists():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            n = r["billed"] or 0
            total += n
            key = r["purpose"].split(":")[0] or "（未注明）"
            by[key] = by.get(key, 0) + n
    return {"total": total, "by": by, "left": FREE - total, "cost": max(total - FREE, 0) * PRICE}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1] == "用量":
        u = usage()
        for k, v in u["by"].items():
            print(f"{k}\t{v}")
        print(f"合计 {u['total']} 字，免费额度还剩 {u['left']} 字，超出部分 {u['cost']:.2f} 元")
        raise SystemExit
    text = sys.argv[1]
    speaker = sys.argv[2] if len(sys.argv) > 2 else SPEAKER
    audio, end = synth(text, speaker, purpose="命令行")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{speaker}-{time.strftime('%Y%m%d-%H%M%S')}.mp3"
    path.write_bytes(audio)
    print(f"{path}  {len(audio)} 字节  {end['secs']:.1f} 秒  用量 {end.get('usage')}  logid {end['logid']}")
