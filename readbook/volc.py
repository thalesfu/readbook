"""火山引擎豆包语音合成 2.0（新版控制台，HTTP Chunked 单向流式 V3）。

凭证只从环境变量 VOLC_TTS_API_KEY 读，代码里不写、不打印。Windows 上新设的用户环境变量，
已经开着的进程看不到，所以环境里没有时再去注册表 HKCU\\Environment 里找。
只发文字，不上传任何原声。

    uv run python -m readbook.volc "要合成的一句话" [音色]   # 写到 out/volc/
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


def synth(text: str, speaker: str = SPEAKER, fmt: str = "mp3", sample_rate: int = 24000) -> tuple[bytes, dict]:
    """合成一段文字，返回 (音频字节, 结束包)。结束包里有 usage（计费字数）。"""
    body = {
        "user": {"uid": "readbook"},
        "req_params": {
            "text": text,
            "speaker": speaker,
            "audio_params": {"format": fmt, "sample_rate": sample_rate},
        },
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
        "Content-Type": "application/json",
        "X-Api-Key": api_key(),
        "X-Api-Resource-Id": RESOURCE,
        "X-Api-Request-Id": str(uuid.uuid4()),
        "X-Control-Require-Usage-Tokens-Return": "text_words",
    })
    audio = bytearray()
    with urllib.request.urlopen(req, timeout=60) as resp:
        logid = resp.headers.get("X-Tt-Logid")
        for line in resp:
            if not line.strip():
                continue
            msg = json.loads(line)
            if msg.get("code") == 20000000:
                return bytes(audio), {**msg, "logid": logid}
            if msg.get("code") not in (0, None):
                raise RuntimeError(f"合成失败：{msg.get('code')} {msg.get('message')}（logid {logid}）")
            if msg.get("data"):
                audio += base64.b64decode(msg["data"])
    raise RuntimeError(f"没有收到结束包（logid {logid}）")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    text = sys.argv[1]
    speaker = sys.argv[2] if len(sys.argv) > 2 else SPEAKER
    t0 = time.time()
    audio, end = synth(text, speaker)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{speaker}-{time.strftime('%Y%m%d-%H%M%S')}.mp3"
    path.write_bytes(audio)
    print(f"{path}  {len(audio)} 字节  {time.time() - t0:.1f} 秒  用量 {end.get('usage')}  logid {end['logid']}")
