"""《白色相簿2》的结构捷径：直接读游戏剧本，不解析 epub。

解包和字节码遍历复用 gameresource 仓库里的工具（lfpak / lscr / daily），
这里只负责按「路线 × 日期」取出一天的台词，并把中文、日文、语音文件对齐。
"""

import re
import sys
from dataclasses import asdict, dataclass

from .config import GAME, TOOLS

sys.path.insert(0, str(TOOLS))

from daily import ROUTES, walk  # noqa: E402
from lfpak import Pak  # noqa: E402
from lscr import TAG_RUBY, render  # noqa: E402

# 原始文本的首字符决定结构类别（render 之前判断，之后引号已经被换掉）
KIND_BY_LEAD = {"「": "speech", "『": "quote"}


@dataclass
class Line:
    idx: int
    sid: str
    kind: str          # speech / quote / plain：看原文开头是「、『 还是都不是
    speaker: str       # 游戏里 WN 给的名字；空串 = 旁白
    text: str          # 中文，读正文不读注音，排版换行已合并
    jp: str            # 日文原文，同样处理
    voice: str | None  # 语音文件名，如 2024_0059_11.OGG

    @property
    def voice_code(self) -> str | None:
        return self.voice.rsplit("_", 1)[1].split(".")[0] if self.voice else None


def clean(raw: str) -> str:
    """注音只留正文，\\n 合并，\\k 分页换成换行。"""
    t = render(TAG_RUBY.sub(r"\1", raw))
    return "\n".join("".join(p.split("\\n")).strip() for p in t.split("\\k") if p.strip())


def strip_quotes(text: str) -> str:
    return re.sub(r"^[“‘](.*)[”’]$", r"\1", text, flags=re.S)


def load_day(route: str, date: tuple[int, int, int]) -> list[Line]:
    """取出某条路线上某一天的全部台词，日期切分规则和 daily.py 一致。"""
    chain = next(c for folder, _, c in ROUTES if folder == route)
    cn, jp = Pak(GAME / "ck-gal.pak"), Pak(GAME / "script.pak")
    out: list[Line] = []
    cur = None
    for sid in chain:
        jp_lines = [ev for ev in walk(jp, sid, "cp932") if ev[0] == "line"]
        n = 0
        for ev in walk(cn, sid, "cn"):
            if ev[0] == "date":
                cur = ev[1]
                continue
            _, speaker, raw, voice = ev
            j = jp_lines[n]
            n += 1
            assert j[3] == voice, f"{sid} 第 {n} 行中日语音不一致"
            if cur != date:
                continue
            kind = KIND_BY_LEAD.get(raw.lstrip()[:1], "plain")
            out.append(Line(len(out), sid, kind, speaker, clean(raw), clean(j[2]), voice))
    return out


if __name__ == "__main__":
    import json

    sys.stdout.reconfigure(encoding="utf-8")  # Windows 上管道默认是 GBK
    route, day = sys.argv[1], tuple(int(x) for x in sys.argv[2].split("-"))
    for line in load_day(route, day):
        print(json.dumps(asdict(line), ensure_ascii=False))
