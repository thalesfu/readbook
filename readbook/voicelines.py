"""原作声线库（生产 04 号票）：把全书的语音编号 × 角色列成声线，写 books/wa2/voices/声线.yaml。

规则见合成规格第四节"声线库"：
- 来源：gameresource 产出的 语音对照.tsv。只用 CN 部分（IC 部分是序章的日文版，语音文件和 CN 重复），
  同一个语音文件只算一次；
- 同一个角色的不同写法按 ALIASES 合并，主写法取 CN 里句数多的。说话人空着或写「？？？」的，是还没报出名字的
  本人（序章「邻座的女生」之前的和纱、初登场的曜子），算进这个编号的主角色（台词最多的那个）；
- 98、99 号混着好几个声优，不做声线；
- 每个（编号, 角色）先算一条声线。可用原声（1.5–12 秒、测得出音高和语速，和 rawref.pool 一样）不到 MIN_CLIPS
  句的，归到主角色的声线；
- 同一个声优的声线（同编号各自成线的角色，加上一个人配两个编号的）两两互认：每句可用原声离哪条声线的原声
  中心更近（CAM++；自己那条的中心去掉这一句），任一边低于 MERGE_BELOW 就列出来，用户听过再定合不合并，
  定了的写进 MERGED；
- 音色门槛：evaluate.real_clips 按这条声线的语音文件取原声，evaluate.voice_gate 的 center_p5；
- 性别见 MALE；
- 音色锚点（生产 05 号票）：rawanchor.make_lines 生成、登记进来的 ANCHOR_KEYS 几项，build 重跑时原样留着。

声线.yaml 只记语音文件名和量出来的数，原声不进 git。

    uv run python -m readbook.voicelines scan     # 量全书原声的时长、音高、语速，缓存在 out/voicelines/（只用 CPU）
    uv run python -m readbook.voicelines build    # 互认和音色门槛（CAM++，用 GPU），写 声线.yaml
    uv run python -m readbook.voicelines listen   # 互认低于 MERGE_BELOW 的各取几句原声拼成试听，放 out/voicelines/试听/
"""

import csv
import json
import random
import sys
from functools import cache
from itertools import combinations
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from . import moods, qc, wa2
from .evaluate import real_clips, voice_gate
from .trial import BOOK, OUT
from .voices import SR, VoicePak

sys.path.insert(0, str(wa2.TOOLS))
from daily import walk  # noqa: E402
from lfpak import Pak  # noqa: E402

TABLE = wa2.TOOLS.parent / "产出" / "语音对照.tsv"
VOICES = BOOK / "voices"
LINES_YAML = VOICES / "声线.yaml"
ANCHOR_KEYS = ("锚点", "锚点文本", "锚点像原声", "目标音高", "目标语速")  # rawanchor.make_lines 登记的
CACHE = OUT / "voicelines"
SHARED = ("98", "99")   # 多个龙套共用的编号
UNNAMED = ("", "？？？")  # 还没报出名字的本人
MIN_CLIPS = 20          # 可用原声不到这么多句，归主角色的声线
MERGE_BELOW = 0.90      # 互认准确率低于它，给用户听
RECOG_N = 300           # 互认时每条声线最多取多少句可用原声（固定种子随机）

# 男声的编号。按声优；查不到声优的（17、18、26、29、37）按角色定，再拿 CAM++ 声纹离男女两组中心的远近核对。
# 音高分不开男女：孝宏（男）中位 222 Hz，榛名（女）199 Hz。
MALE = {"00", "10", "12", "13", "15", "17", "18", "20", "21", "27", "28", "30", "31", "32", "36", "37"}

# 同一个角色在 CN 剧本里的其他写法 → 主写法
ALIASES = {
    "かずさ": "和纱", "依緒": "依绪", "親志": "亲志", "耀子": "曜子",
    "雪菜的父亲": "雪菜父亲", "雪菜的母亲": "雪菜母亲", "雪菜的母亲＆孝宏": "雪菜母亲＆孝宏",
    "亜子": "亚子", "美穂子": "美穰子",
}

# 声优（编号 → 名字）。PC 版没公开声优，这里是 PS3 版、日文维基、VNDB 的署名（VNDB 把同一批人记在 PC 版上）；
# 查不到的不写，声线名用「编号号」。20 和 36 是同一个人。
ACTORS = {
    "00": "水島大宙", "01": "生天目仁美", "02": "米澤円", "03": "永田依子", "04": "羽吹梨里", "05": "浅川悠",
    "10": "寺島拓篤", "11": "中上育実", "12": "杉山紀彰", "13": "最上嗣生", "14": "小野涼子", "15": "梶裕貴",
    "16": "夏樹リオ", "19": "柏木美優", "20": "三戸耕三", "21": "渡辺浩司", "22": "小暮英麻", "23": "一杉佳澄",
    "24": "伊東久美子", "25": "澄田まりや", "27": "白熊寛嗣", "28": "室元気", "30": "浜田賢二", "31": "村田太志",
    "32": "梶川翔平", "33": "山川琴美", "34": "石塚さより", "35": "上川ゆう", "36": "三戸耕三",
}

# 用户听过定下的合并：(编号, 角色) → 并进同编号哪个角色的声线
MERGED: dict[tuple[str, str], str] = {
    ("04", "雪音"): "千晶",  # 互认 0.817 / 0.969；雪音是千晶在舞台剧里演的角色（2026-10-30 用户："好"）
    ("28", "吉田"): "和希",  # 互认 0.750 / 0.590；和希是吉田在舞台剧里演的角色（2026-10-30 用户："好"）
}


# ---------- 数据 ----------

def code_of(voice: str) -> str:
    return voice.rsplit("_", 1)[1].split(".")[0]


@cache
def table() -> dict[str, tuple[str, str]]:
    """语音文件 → (剧本, 说话人主写法)：CN 部分，每个文件只取第一次出现，去掉 98、99 号和语音包里没有的
    （LF3 的两句）。"""
    have = VoicePak().index
    out = {}
    with TABLE.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            f = r["语音文件"]
            if r["章节"] != "CN" or f in out or f not in have or code_of(f) in SHARED:
                continue
            out[f] = (r["剧本"], ALIASES.get(r["说话人"], r["说话人"]))
    return out


def roles(tab: dict[str, tuple[str, str]]) -> dict[str, dict[str, list[str]]]:
    """{编号: {角色: [语音文件]}}，主角色在前，其余按句数从多到少；没报名字的算进主角色。"""
    by = {}
    for f, (_, who) in sorted(tab.items()):
        by.setdefault(code_of(f), {}).setdefault(who, []).append(f)
    out = {}
    for code in sorted(by):
        named = sorted((n for n in by[code] if n not in UNNAMED), key=lambda n: -len(by[code][n]))
        main = named[0]
        out[code] = {n: by[code][n] for n in named}
        out[code][main] = sorted(out[code][main] + [f for n in UNNAMED for f in by[code].get(n, [])])
    return out


def jp_text(sids: set[str]) -> dict[str, str]:
    """语音文件 → 日文台词（和 bakeoff.voiced_lines 同样清理）。"""
    jp = Pak(wa2.GAME / "script.pak")
    out = {}
    for sid in sorted(sids):
        for ev in walk(jp, sid, "cp932"):
            if ev[0] == "line" and ev[3]:
                out.setdefault(ev[3], wa2.strip_quotes(wa2.clean(ev[2]).replace("\n", "")))
    return out


# ---------- 量原声 ----------

def scan():
    """每个语音文件的时长；1.5–12 秒的再量音高和日语语速。缓存在 out/voicelines/clips.json，可以断点续跑。"""
    path = CACHE / "clips.json"
    have = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    tab = table()
    todo = [f for f in tab if f not in have]
    if not todo:
        return have
    jp = jp_text({tab[f][0] for f in todo})
    vp = VoicePak()
    CACHE.mkdir(parents=True, exist_ok=True)
    for k, f in enumerate(todo, 1):
        a = vp.audio(f)
        dur = len(a) / SR
        rec = {"dur": round(dur, 2)}
        if 1.5 <= dur <= 12:
            fz, rt = qc.f0_median(a), qc.rate(a, jp.get(f, ""))
            rec |= {"f0": round(fz, 1) if fz else None, "rate": round(rt, 3) if rt else None}
        have[f] = rec
        if k % 2000 == 0 or k == len(todo):
            path.write_text(json.dumps(have, ensure_ascii=False), encoding="utf-8")
            print(f"{k}/{len(todo)}", flush=True)
    return have


def usable(clips: dict, f: str) -> bool:
    c = clips.get(f, {})
    return bool(c.get("f0") and c.get("rate"))


# ---------- 声线 ----------

def make_lines(rs: dict[str, dict[str, list[str]]], clips: dict) -> list[dict]:
    """每个（编号, 角色）一条；可用原声不够或用户定了合并的，并进主角色（或 MERGED 指定的角色）那条。"""
    lines = []
    for code, by in rs.items():
        main = next(iter(by))
        own = {n: [n] for n in by if n == main or (sum(usable(clips, f) for f in by[n]) >= MIN_CLIPS
                                                   and (code, n) not in MERGED)}
        for n in by:
            if n not in own:
                own[MERGED.get((code, n), main)].append(n)
        for n, members in own.items():
            files = sorted(f for m in members for f in by[m])
            lines.append({"code": code, "main": n, "roles": {m: len(by[m]) for m in members}, "files": files})
    return lines


def line_name(line: dict) -> str:
    """「声优-角色…」：主角色，加上用户定了合并进来的角色（可用原声够数、自己能成线的）；
    原声不够、借用这条声线的小角色只记在「角色」里。"""
    big = [line["main"]] + [n for n in line["roles"] if (line["code"], n) in MERGED]
    return f"{ACTORS.get(line['code'], line['code'] + '号')}-{'，'.join(big)}"


class Embeddings:
    """语音文件 → CAM++ 向量，缓存在 out/voicelines/emb.npz。"""

    def __init__(self, vp: VoicePak):
        from .speaker import SpeakerSim

        self.vp, self.sp, self.path = vp, SpeakerSim(), CACHE / "emb.npz"
        self.have = dict(np.load(self.path)) if self.path.exists() else {}
        self.new = 0

    def __call__(self, f: str, a: np.ndarray | None = None) -> np.ndarray:
        if f not in self.have:
            self.have[f] = self.sp.embed_audio(self.vp.audio(f) if a is None else a, SR)
            self.new += 1
            if self.new % 2000 == 0:
                self.save()
        return self.have[f]

    def save(self):
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, **self.have)


def recog_files(line: dict, clips: dict) -> list[str]:
    fs = [f for f in line["files"] if usable(clips, f)]
    random.Random(0).shuffle(fs)
    return sorted(fs[:RECOG_N])


def recognize(ea: np.ndarray, eb: np.ndarray) -> tuple[float, float]:
    """两条声线互认：每句离哪条的原声中心更近，自己那条的中心去掉这一句。返回 (a 认对的比例, b 认对的比例)。"""
    def hit(own, other):
        rest = own.sum(axis=0)[None] - own
        s_own = np.einsum("ij,ij->i", own, rest) / np.linalg.norm(rest, axis=1)
        c = other.sum(axis=0)
        return float(np.mean(s_own > own @ (c / np.linalg.norm(c))))
    return hit(ea, eb), hit(eb, ea)


def build():
    clips = scan()
    lines = make_lines(roles(table()), clips)
    by_code = {}
    for line in lines:
        by_code.setdefault(line["code"], []).append(line)
    vp = VoicePak()
    emb = Embeddings(vp)
    old = load() if LINES_YAML.exists() else {}
    out = {}
    for line in lines:
        code, files = line["code"], line["files"]
        use = [f for f in files if usable(clips, f)]
        f0 = [clips[f]["f0"] for f in use]
        # 编号只有一条声线时和 evaluate 一样取整个编号，现有四人的门槛不变（同 gate_files）
        E = np.array([emb(f, a) for f, a in real_clips(vp, code, files=None if len(by_code[code]) == 1 else files)])
        _, gate = voice_gate(E) if len(E) > 1 else (None, {"n": len(E), "center_p5": None})
        med = float(np.median(f0)) if f0 else None
        out[line_name(line)] = {
            "声优": ACTORS.get(code), "编号": code, "角色": line["roles"],
            "原声句数": len(use), "性别": "男" if code in MALE else "女",
            "音高中位": round(med, 1) if med else None,
            "日语语速中位": round(float(np.median([clips[f]["rate"] for f in use])), 2) if use else None,
            "音色门槛": round(gate["center_p5"], 3) if gate["center_p5"] is not None else None,
            "门槛句数": gate["n"],
            **{k: v for k, v in old.get(line_name(line), {}).items() if k in ANCHOR_KEYS},
        }
        print(line_name(line), out[line_name(line)]["原声句数"], out[line_name(line)]["音色门槛"], flush=True)
    by_actor = {}  # 同一个声优的声线两两互认：同编号的，加上一个人配两个编号的（20 和 36）
    for line in lines:
        by_actor.setdefault(ACTORS.get(line["code"], line["code"]), []).append(line)
    for ls in by_actor.values():
        for a, b in combinations(ls, 2):
            ea = np.array([emb(f) for f in recog_files(a, clips)])
            eb = np.array([emb(f) for f in recog_files(b, clips)])
            ha, hb = recognize(ea, eb)
            out[line_name(a)].setdefault("互认", {})[line_name(b)] = round(ha, 3)
            out[line_name(b)].setdefault("互认", {})[line_name(a)] = round(hb, 3)
            print("互认", line_name(a), f"{ha:.3f}", line_name(b), f"{hb:.3f}", flush=True)
    emb.save()
    save(out)


def save(lines: dict):
    """写 声线.yaml：文件头说明，加上写法和声线两块。"""
    aliases = {}
    for alt, main in ALIASES.items():
        aliases.setdefault(main, []).append(alt)
    head = ("# 原作声线库（生产 04 号票），规则见 readbook/voicelines.py 和合成规格第四节。只记量出来的数，原声不进 git；\n"
            "# 每条声线的语音文件按 语音对照.tsv 现取（voicelines.files）。98、99 号混着几个声优，不做声线。\n"
            "# - 声线名「声优-角色…」，查不到声优的写「编号号」。PC 版没公开声优，按 PS3 版、日文维基、VNDB 的署名；\n"
            "# - 角色：这条声线配的角色和全书句数（按语音文件去重），第一个是主角色。说话人空着或「？？？」的算进主角色；\n"
            f"#   可用原声不到 {MIN_CLIPS} 句的角色借主角色的声线；\n"
            "# - 原声句数：可用原声（1.5–12 秒、测得出音高和语速）；音高中位（Hz）、日语语速中位（字/秒）按它算；\n"
            "# - 性别：按声优，查不到的按角色，用声纹核对过（voicelines.MALE；音高分不开男女）；\n"
            "# - 音色门槛：evaluate.voice_gate 的 center_p5，原声按 evaluate.real_clips 取（1.5–15 秒，最多 300 句，种子 0）；\n"
            f"# - 互认：和同一个声优的其他声线互相认人时，这条声线的原声认对的比例（最多 {RECOG_N} 句可用原声）；\n"
            "# - 音色锚点（生产 05 号票，rawanchor.make_lines，规则见 readbook/rawanchor.py 和合成规格第四节）：\n"
            "#   锚点＝锚点录音，路径相对 books/wa2/voices/；锚点文本＝它念的字；锚点像原声＝它和这条声线原声中心的\n"
            "#   相似度（整段，CAM++）；目标音高（Hz）、目标语速（字/秒）是逐句质检的目标。挑选的明细在 锚点.yaml。\n")
    LINES_YAML.write_text(head + yaml.safe_dump({"写法": aliases, "声线": lines}, allow_unicode=True,
                                               sort_keys=False, width=200), encoding="utf-8", newline="\n")
    print(LINES_YAML)


# ---------- 给别的模块用 ----------

def load() -> dict:
    return yaml.safe_load(LINES_YAML.read_text(encoding="utf-8"))["声线"]


def gate_files(name: str, lines: dict | None = None) -> list[str] | None:
    """音色门槛取原声的范围（evaluate.real_clips 的 files）：编号只有一条声线时取整个编号（None，和 evaluate
    一样，现有四人的门槛不变），拆成几条的按这条声线的语音文件取。"""
    lines = lines or load()
    code = lines[name]["编号"]
    return None if sum(v["编号"] == code for v in lines.values()) == 1 else files(name, lines)


def anchor(name: str, lines: dict | None = None) -> Path:
    """一条声线的音色锚点录音（整段；IndexTTS 只取前 15 秒，见 index_passage.anchor_clip）。"""
    lines = lines or load()
    return VOICES / lines[name]["锚点"]


def line_of(code: str, speaker: str, lines: dict | None = None) -> str | None:
    """某个编号下某个说话人用哪条声线：角色自己的，没有就用这个编号的主角色的；98、99 号返回 None。"""
    if code in SHARED:
        return None
    lines = lines or load()
    speaker = ALIASES.get(speaker, speaker)
    mine = [n for n, v in lines.items() if v["编号"] == code]
    return next((n for n in mine if speaker in lines[n]["角色"]), mine[0] if mine else None)


def files(name: str, lines: dict | None = None) -> list[str]:
    """一条声线的全部语音文件。"""
    lines = lines or load()
    v = lines[name]
    main = next(n for n, x in lines.items() if x["编号"] == v["编号"]) == name  # 这个编号的第一条是主角色的
    return sorted(f for f, (_, who) in table().items() if code_of(f) == v["编号"]
                  and (who in v["角色"] or (main and who in UNNAMED)))


# ---------- 试听 ----------

def listen(per=8, seed=0):
    """互认低于 MERGE_BELOW 的每对声线，各取 per 句 3–8 秒的可用原声拼一段，放 out/voicelines/试听/。"""
    clips = scan()
    lines = load()
    vp = VoicePak()
    dst = CACHE / "试听"
    dst.mkdir(parents=True, exist_ok=True)
    pairs = {tuple(sorted((a, b))) for a, v in lines.items() for b, acc in v.get("互认", {}).items()
             if min(acc, lines[b]["互认"][a]) < MERGE_BELOW}
    for pair in sorted(pairs):
        for n in pair:
            fs = [f for f in files(n, lines) if usable(clips, f) and 3 <= clips[f]["dur"] <= 8]
            random.Random(seed).shuffle(fs)
            pick = sorted(fs[:per])
            path = dst / f"{lines[n]['编号']}-{n}.wav"
            sf.write(path, moods.concat([vp.audio(f) for f in pick], SR, gap=0.6), SR)
            print(path, pick, flush=True)


if __name__ == "__main__":
    {"scan": scan, "build": build, "listen": listen}[sys.argv[1]]()
