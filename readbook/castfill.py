"""补齐 cast.yaml（生产 06 号票）：全书开口的角色都进角色表。

规则：
- 范围：daily.ROUTES 里的剧本，就是每日对话那本书（5xxx、LF3、catch 不在书里）。说话人和语音文件按语音对照表的
  CN 部分，日期和台词按 daily.walk 现读，两边逐行核对；
- 句数：有语音的按语音文件去重（回忆里重放的同一句只算一次），没语音的按行算；
- 写法合并：voicelines.ALIASES，加上这里的 ALIASES（同一个人：报名字之前的出场、日文原名一样只是没翻译的写法）。
  对着日文原名和上下文核对过；日文原名一样、但出现在不同场景的泛称（男生１／男学生１／男子学生１）不合并；
- 档位：主要 ≥500 句，配角 30–499 句，龙套 <30 句。合称单列；
- code：只填专属编号，就是声线库里这个编号的主角色。一个编号只给一个角色，因为 scriptize 按编号认人；
- 性别：有专属编号的按声线库；98 号是女声龙套、99 号是男声龙套（同一个名字里「ウェイトレス」是 98、「ウェイター」是
  99）；一个名字里男女都有的，按句数多的，列进 MIXED_GENDER 给 07 号票；
- voiceline：名字在声线库某条声线名里的（自己的声线，或用户定了合并进去的）填那条；其余留给 07 号票；
- 年龄、描述、孝宏和朋的性格卡：claude -p 看剧本起草，只发文字，用户抽查；
- pan：见 pan()。

    uv run python -m readbook.castfill count    # 数全书说话人，写 out/castfill/speakers.json，打印合称
    uv run python -m readbook.castfill draft    # claude -p 起草年龄、描述和两张性格卡，缓存在 out/castfill/
    uv run python -m readbook.castfill write    # 补进 cast.yaml：已有角色只补缺的字段
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from functools import cache

import yaml

from . import voicelines as vl
from . import wa2
from .trial import BOOK, OUT

sys.path.insert(0, str(wa2.TOOLS))
from daily import ROUTES, walk  # noqa: E402
from lfpak import Pak  # noqa: E402

CAST = BOOK / "cast.yaml"
CACHE = OUT / "castfill"
FEMALE_MOB, MALE_MOB = "98", "99"
TIERS = ((500, "主要"), (30, "配角"), (0, "龙套"))

# 同一个人，声线库的写法合并之外又查出来的
ALIASES = {
    # 报名字之前的出场：同一个声优，同一场里接着就叫出了名字
    "邻座的女生": "和纱",        # 序章 1002–1004，01 号
    "和泉": "千晶",              # 2001，春希叫她「和泉」，04 号
    "女生２柳原": "朋",          # 1008，19 号，下一句「比朋更加想要显眼」
    "男子店员１": "本田",        # 2005，春希叫他「本田」，21 号
    "男性编辑部员１": "浜田", "男性编辑部员２": "松冈",  # 2001，30、31 号，接着就叫出名字
    "男性编辑部员３": "木崎", "女性编辑部员１": "铃木",  # 「木崎那家伙咋了」正接在编辑部员３跑开之后；33 号
    "指导老师": "诹访老师",      # 1005，18 号，班主任叫他「诹访老师」
    "女子学生７": "雪菜",        # 2001，02 号，「我…无法唱歌」，同伴接着喊「走吧，雪菜！」
    # 日文原名一样，中文没翻译或用了日文字形
    "女性記者": "女记者", "編集長": "主编", "車内广播": "车内广播", "男性劆団員４": "男性团员４", "校長": "校长",
    # 日文原名一样，同一场
    "空港係員": "工作人员", "中年男": "中年男性", "男性录音职员": "男性录音室职员", "女性录音职员": "女性录音室职员",
}

# 集体说的（没有「＆」的合称）
COLLECTIVE = {"全员", "全部人", "全体", "店员们", "众店员", "乘客们", "女店员们", "女子店员们", "小木曽家的众人"}
UNNAMED = vl.UNNAMED  # 空着或「？？？」：剧本化按上下文和编号认人，不进角色表


def main_name(raw: str) -> str:
    n = vl.ALIASES.get(raw, raw)
    return ALIASES.get(n, n)


def is_combo(name: str) -> bool:
    return "＆" in name or name in COLLECTIVE


# ---------- 数 ----------

@cache
def book_lines() -> list[dict]:
    """全书逐行：剧本、序号、日期、说话人（原写法）、语音文件、台词。日期按 daily.py 切：路线里没有日期标记的（序章）
    一个剧本算一天，第一个日期标记之前的算「日期不明」。"""
    table = {}
    with (wa2.TOOLS.parent / "产出" / "语音对照.tsv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["章节"] == "CN":
                table[r["剧本"], int(r["序号"])] = (r["说话人"], r["语音文件"] or None)
    cn = Pak(wa2.GAME / "ck-gal.pak")
    out = []
    for folder, _, chain in ROUTES:
        evs = {sid: list(walk(cn, sid, "cn")) for sid in chain}
        dated = any(e[0] == "date" for es in evs.values() for e in es)
        cur = None
        for sid in chain:
            n = 0
            for e in evs[sid]:
                if e[0] == "date":
                    cur = e[1]
                    continue
                _, who, raw, voice = e
                assert table[sid, n] == (who, voice), f"{sid} 第 {n} 行和语音对照表对不上"
                day = (f"{cur[0]}-{cur[1]:02d}-{cur[2]:02d}" if cur else "日期不明") if dated else f"剧本{sid}"
                out.append({"route": folder, "sid": sid, "n": n, "day": day, "raw": who, "voice": voice,
                            "text": wa2.clean(raw).replace("\n", "")})
                n += 1
    return out


def speakers() -> dict[str, dict]:
    """主写法 → 句数、语音编号、没语音的句数、写法、出场的剧本和天数。"""
    seen, out = set(), {}
    for x in book_lines():
        if x["raw"] in UNNAMED:
            continue
        if x["voice"]:
            if x["voice"] in seen:
                continue
            seen.add(x["voice"])
        s = out.setdefault(main_name(x["raw"]), {"lines": 0, "codes": Counter(), "unvoiced": 0, "spellings": Counter(),
                                                 "sids": Counter(), "days": set()})
        s["lines"] += 1
        if x["voice"]:
            s["codes"][vl.code_of(x["voice"])] += 1
        else:
            s["unvoiced"] += 1
        s["spellings"][x["raw"]] += 1
        s["sids"][x["sid"]] += 1
        s["days"].add(f"{x['route']}/{x['day']}")
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["lines"]))


def owners() -> dict[str, str]:
    """专属编号 → 主角色（声线库里这个编号的第一条声线的第一个角色）。"""
    out = {}
    for v in vl.load().values():
        out.setdefault(v["编号"], next(iter(v["角色"])))
    return out


def own_line(name: str, lines: dict) -> str | None:
    """名字在哪条声线的名字里（「声优-角色1，角色2」）。"""
    return next((n for n in lines if name in n.split("-", 1)[1].split("，")), None)


def tier(n: int) -> str:
    return next(t for lo, t in TIERS if n >= lo)


def gender(s: dict, code: str | None) -> tuple[str | None, bool]:
    """(性别, 男女混着)：专属编号按声线库；其余按语音编号投票，98 女、99 男、别人的编号按那个编号；没语音的不算。"""
    if code:
        return ("男" if code in vl.MALE else "女"), False
    votes = Counter()
    for c, k in s["codes"].items():
        if k:
            votes["女" if c == FEMALE_MOB else "男" if c == MALE_MOB or c in vl.MALE else "女"] += k
    if not votes:
        return None, False
    return votes.most_common(1)[0][0], len(votes) > 1


def count():
    sp = speakers()
    CACHE.mkdir(parents=True, exist_ok=True)
    dump = {n: {**s, "codes": dict(s["codes"]), "spellings": dict(s["spellings"]), "sids": dict(s["sids"]),
                "days": sorted(s["days"])} for n, s in sp.items()}
    (CACHE / "speakers.json").write_text(json.dumps(dump, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(sp)} 个名字（合并写法后，不含空着和「？？？」），合称 {sum(is_combo(n) for n in sp)} 个")
    print("档位：", Counter("合称" if is_combo(n) else tier(s["lines"]) for n, s in sp.items()))
    for x in book_lines():
        if is_combo(main_name(x["raw"])):
            print(x["route"], x["day"], x["sid"], x["n"], x["raw"], x["voice"], x["text"])


# ---------- pan ----------

SLOTS = (-0.2, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.2)  # 台词都偏一点，不放正中（正中是旁白）
NEAR = 10     # 同一个剧本里相隔不到这么多行的两句台词，算同场
APART = 0.1   # 同场的两个角色 pan 至少错开这么多，不到的按差多少算代价（0.15 时同一格的多一倍多）


def together() -> dict[tuple[str, str], int]:
    """(角色, 角色) → 同场的次数：同一个剧本里两句不同人的台词相隔不到 NEAR 行，记一次。合称和没名字的不算。"""
    by_sid = defaultdict(list)
    for x in book_lines():
        n = main_name(x["raw"])
        if n and x["raw"] not in UNNAMED and not is_combo(n):
            by_sid[x["sid"]].append((x["n"], n))
    w = Counter()
    for rows in by_sid.values():
        for i, (ni, a) in enumerate(rows):
            for nj, b in rows[i + 1:]:
                if nj - ni >= NEAR:
                    break
                if a != b:
                    w[a, b] += 1
                    w[b, a] += 1
    return w


def pan(names: list[str], fixed: dict[str, float]) -> dict[str, float]:
    """新角色的 pan：在 SLOTS 里挑和同场角色撞得最轻的一格。
    代价 = Σ 同场次数 × max(0, 1 − 两人相差 / APART)；一样轻的挑离正中近的，再挑左边。已有角色的 pan 不动。
    先按句数从多到少一个个挑（只看已经定了的），再轮流让每个新角色按其余所有人重挑，直到没人再换。"""
    w = defaultdict(dict)
    for (a, b), k in together().items():
        w[a][b] = k
    out = dict(fixed)

    def best(n):
        def cost(s):
            return sum(k * max(0.0, 1 - abs(s - out[o]) / APART) for o, k in w[n].items() if o in out and o != n)
        return min(SLOTS, key=lambda s: (round(cost(s), 6), abs(s), s))

    for n in names:
        out[n] = best(n)
    for _ in range(50):
        moved = 0
        for n in names:
            s = best(n)
            moved += s != out[n]
            out[n] = s
        if not moved:
            break
    return {n: out[n] for n in names}


# ---------- 起草：年龄、描述、性格卡 ----------

DRAFT_RULES = """\
你在给《白色相簿2》有声书补角色表。时间线：第一部序章是 2007 年峰城大附属高中三年级；第二部是 2010–2011 年，春希他们上大学；
第三部是 2012–2013 年，已经工作。

下面每个角色给了：档位和句数、性别（按原作配音定的；写「？」的要你判断）、出场的路线和日期、几句台词（每句前面一行是上下文）。

给每个角色写：
- gender：男 或 女。给了的照抄；
- age：大概的年龄，整数。学生按年级估，社会人按身份估；
- desc：一句人物与声音描述，给配音挑声线用：
  - 先写这是谁：身份、和主角们的关系、在什么场合出场；
  - 名字是泛称（店员、男子学生１、车站广播），而且出现在不同场景的，写成泛称，注明是各处的，不要当成一个人写身世；
  - 再写性格或说话的样子；
  - 最后写声音：年龄感、音色、语气，比如「中年男声，粗一点，爽朗」；
  - 配角 40–70 字，龙套 20–40 字；
  - 只依据给出的台词和你确知的原作设定，拿不准的不要编。

已有的例子：
- 雪菜：小木曾雪菜，开朗温柔、受欢迎的优等生，会唱歌；表面明亮，内心敏感。年轻女声，明亮甜美，偏高。
- 工作人员２：摄制组工作人员，招呼大家休息。

按 schema 返回 characters，每个角色一条，name 照抄。
"""

DRAFT_SCHEMA = {
    "type": "object",
    "properties": {"characters": {"type": "array", "items": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "gender": {"type": "string", "enum": ["男", "女"]},
                       "age": {"type": "integer"}, "desc": {"type": "string"}},
        "required": ["name", "gender", "age", "desc"]}}},
    "required": ["characters"],
}

CARD_RULES = """\
你在给《白色相簿2》有声书补主要角色的性格卡。卡是给逐句定表演设置的 LLM 看的，格式和已有的一样：
- age：大概的年龄（第二部，2010 年时），整数；
- desc：一句人物与声音描述（身份、性格，再写声音），60–90 字；
- card：
  - 性格：一两句；
  - 情绪外露：情绪怎么表现在声音里，开心、生气、难过、慌乱时各是什么样；
  - 声音：音色和轻声时的样子。

已有的例子：
{examples}

下面是 {name} 的资料和全书各处的台词（每句前面一行是上下文）。只依据台词和你确知的原作设定。按 schema 返回。
"""

CARD_SCHEMA = {
    "type": "object",
    "properties": {"age": {"type": "integer"}, "desc": {"type": "string"}, "card": {
        "type": "object",
        "properties": {"性格": {"type": "string"}, "情绪外露": {"type": "string"}, "声音": {"type": "string"}},
        "required": ["性格", "情绪外露", "声音"]}},
    "required": ["age", "desc", "card"],
}

CARD_FOR = ("孝宏", "朋")
BATCH = 20


def samples(name: str, k: int) -> list[str]:
    """这个角色的 k 句台词，各剧本轮流取，带前一行做上下文。"""
    rows = book_lines()
    by_sid = defaultdict(list)
    for i, x in enumerate(rows):
        if main_name(x["raw"]) == name:
            by_sid[x["sid"]].append(i)
    picked, depth = [], 0
    while len(picked) < k and any(depth < len(v) for v in by_sid.values()):
        picked += [v[depth] for v in by_sid.values() if depth < len(v)]
        depth += 1
    out = []
    for i in sorted(picked[:k]):
        prev = rows[i - 1] if i and rows[i - 1]["sid"] == rows[i]["sid"] else None
        if prev:
            out.append(f"    （{prev['raw'] or '旁白'}）{prev['text'][:60]}")
        out.append(f"  - {rows[i]['route']} {rows[i]['day']}：{rows[i]['text'][:80]}")
    return out


def profile(name: str, s: dict, sex: str | None, k: int) -> str:
    days = sorted(s["days"])
    where = "、".join(days[:6]) + (f" 等 {len(days)} 天" if len(days) > 6 else "")
    return "\n".join([f"## {name}", f"- {tier(s['lines'])}，{s['lines']} 句；性别：{sex or '？'}；出场：{where}",
                      "- 台词："] + samples(name, k))


def draft():
    """claude -p 起草：新角色的性别（没语音的）、年龄、描述，孝宏和朋的性格卡。按批缓存，重跑只补没做的。"""
    from concurrent.futures import ThreadPoolExecutor

    from .scriptize import ask_claude

    cast = yaml.safe_load(CAST.read_text(encoding="utf-8"))["characters"]
    sp, own = speakers(), owners()
    code_of_name = {n: c for c, n in own.items()}
    CACHE.mkdir(parents=True, exist_ok=True)
    have = drafts(cards=False)
    todo = [n for n in sp if n not in cast and not is_combo(n) and n not in have]
    first = len(list(CACHE.glob("draft-*.json")))
    jobs = []
    for b in range(0, len(todo), BATCH):
        path = CACHE / f"draft-{first + b // BATCH:02d}.json"
        names = todo[b:b + BATCH]
        body = "\n\n".join(profile(n, sp[n], gender(sp[n], code_of_name.get(n))[0], 20 if tier(sp[n]["lines"]) != "龙套" else 8)
                           for n in names)
        jobs.append((path, names, DRAFT_RULES + "\n" + body, DRAFT_SCHEMA))
    examples = "\n".join(f"### {n}\n- desc：{cast[n]['desc']}\n" + "\n".join(f"- {k}：{v}" for k, v in cast[n]["card"].items())
                         for n in ("小春", "依绪"))
    lines = vl.load()
    for n in CARD_FOR:
        path = CACHE / f"card-{n}.json"
        if path.exists():
            continue
        v = lines[own_line(n, lines)]
        head = f"- 主要角色，{sp[n]['lines']} 句；性别：{v['性别']}；原作声优 {v['声优']}，原声音高中位 {v['音高中位']} Hz"
        jobs.append((path, [n], CARD_RULES.format(examples=examples, name=n) + "\n" + head + "\n- 台词：\n"
                     + "\n".join(samples(n, 60)), CARD_SCHEMA))

    def run(job):
        path, names, prompt, schema = job
        meta = {}
        data = ask_claude(prompt, schema=schema, meta=meta)
        if schema is DRAFT_SCHEMA:
            got = {c["name"]: c for c in data["characters"]}
            missing = [n for n in names if n not in got]
            if missing:
                raise ValueError(f"{path.name} 漏了 {missing}")
            data = [got[n] for n in names]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(path.name, f"{meta.get('duration_ms', 0) / 1000:.0f} 秒", f"${meta.get('total_cost_usd', 0):.2f}", flush=True)

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(run, jobs))


def drafts(cards: bool = True) -> dict[str, dict]:
    """起草的结果：名字 → gender、age、desc（孝宏和朋再加 card，desc 和 age 用性格卡那一遍的）。"""
    out = {}
    for p in sorted(CACHE.glob("draft-*.json")):
        out |= {c["name"]: c for c in json.loads(p.read_text(encoding="utf-8"))}
    for n in CARD_FOR if cards else ():
        out.setdefault(n, {}).update(json.loads((CACHE / f"card-{n}.json").read_text(encoding="utf-8")))
    return out


# ---------- 写 cast.yaml ----------

# 千晶在舞台剧录像里演的角色（剧本 2405），合成规格第四节写明用千晶的声线
PLAYED = {"女主角１": "千晶", "女主角２": "千晶", "女主角３": "千晶", "女主角４": "千晶"}

# 起草以后对着剧本改的
DESC_FIX = {
    "美穰子的母亲": "小春线里矢田美穰子的母亲，女儿闭门不去学校时接待来访的小春，担心女儿能否毕业，感激小春一直照顾她。"
                   "温和客气，忧心忡忡。中年女声，柔和稳重，略显疲惫。",
    # 2005 那三句是中川报名字之前（22 号），2317 那句是别人，按泛称写
    "女子店员１": "春希打工的家庭餐厅的女店员（泛称），热情招呼，爱拖长音、爱八卦。年轻女声，活泼亲切，语尾轻快上扬。",
}

COLLECTIVE_DESC = {
    "全员": "在场的人一起说（干杯）。", "全部人": "在场的人一起说（干杯、道贺）。", "全体": "在场的人一起说（干杯）。",
    "小木曽家的众人": "小木曾家的人一起说（干杯）。",
}

FIELDS = ("tier", "code", "voiceline", "combo", "gender", "age", "pan", "aliases", "desc", "card")
HEAD_OLD = """\
# 《白色相簿2》角色表
#
# tier：主要（≥500 句）/ 配角（30–499 句）/ 龙套（<30 句）
# code：游戏语音编号（VOICE.PAK 文件名最后一段）。有编号的角色复刻；
#       98/99 是多个龙套共用的，不复刻，走 pool 里的设计音色。
# pan：立体声偏移，-1 最左、+1 最右；只作用于台词，旁白和心理活动一律居中。
"""
HEAD_NEW = """\
# 《白色相簿2》角色表。全书开口的角色都在这里（生产 06 号票补齐，规则见 readbook/castfill.py）。
#
# tier：主要（≥500 句）/ 配角（30–499 句）/ 龙套（<30 句），按合并写法后的全书句数；几个人一起说的是「合称」。
# code：专属语音编号（VOICE.PAK 文件名最后一段），只给这个编号的主角色，一个编号一个角色（scriptize 按它认人）。
#       98/99 是多个龙套共用的，不复刻；已有的三个龙套写着 99。
# voiceline：用哪条声线（books/wa2/voices/声线.yaml）。有专属编号的，和声线名里有自己的（榛名、雪音、吉田），
#       照声线库填；千晶演的女主角１～４用千晶的（合成规格第四节）；其余留给 07 号票分。
# combo：合称是谁一起说（06 号票，用户定）。列了名字的：一句里两段引号的拆开，按顺序各念各的；同一句的，每人用
#       自己的声线合成、按各自的 pan 叠在一起。「在场」：剧本化时按上下文列出在场开口的 3–4 人这样叠，
#       列不出人名的（路人一起喊）叠 07 号票配的 3 条龙套声线。
# pan：立体声偏移，-1 最左、+1 最右；只作用于台词，旁白和心理活动一律居中。06 号票起新角色按 castfill.pan 定：
#       ±0.2 以内、不放正中，同场的错开；合称不写，各用各的。
"""


def scalar(v) -> str:
    return yaml.safe_dump(v, allow_unicode=True, width=10 ** 6).removesuffix("\n...\n").strip()


def emit(name: str, d: dict) -> list[str]:
    out = [f"  {name}:"]
    for k in FIELDS:
        if k not in d:
            continue
        v = d[k]
        if k == "card":
            out += ["    card:"] + [f"      {ck}: {scalar(cv)}" for ck, cv in v.items()]
        elif isinstance(v, list):
            out.append(f"    {k}: [{', '.join(scalar(x) for x in v)}]")
        elif k == "code":
            out.append(f'    {k}: "{v}"')
        else:
            out.append(f"    {k}: {scalar(v)}")
    return out


def entries() -> tuple[dict[str, dict], dict[str, dict]]:
    """(已有角色要补的字段, 新角色的全部字段)。"""
    text = CAST.read_text(encoding="utf-8")
    cast = yaml.safe_load(text)["characters"]
    sp, own, lines, dr = speakers(), owners(), vl.load(), drafts()
    code_of_name = {n: c for c, n in own.items()}
    new_names = [n for n in sp if n not in cast and not is_combo(n)]
    pans = pan(new_names, {n: c["pan"] for n, c in cast.items() if "pan" in c})  # 合称没有 pan
    add, new = {}, {}
    for n, s in sp.items():
        aliases = [a for a in s["spellings"] if a != n]
        line = own_line(PLAYED.get(n, n), lines)
        if n in cast:
            c = cast[n]
            d = {}
            if line and "voiceline" not in c:
                d["voiceline"] = line
            more = [a for a in aliases if a not in c.get("aliases", [])]
            if more:
                d["aliases"] = c.get("aliases", []) + more
            if d:
                add[n] = d
            continue
        d = {"tier": "合称" if is_combo(n) else tier(s["lines"])}
        if n in code_of_name:
            d["code"] = code_of_name[n]
        if line:
            d["voiceline"] = line
        if is_combo(n):  # 集体的不写性别：录音虽是 98 或 99 号，在场的人男女都有
            d["combo"] = [main_name(m) for m in n.split("＆")] if "＆" in n else "在场"
        else:
            d["gender"] = gender(s, code_of_name.get(n))[0] or dr[n]["gender"]
            d["age"] = dr[n]["age"]
            d["pan"] = pans[n]
        if aliases:
            d["aliases"] = aliases
        if is_combo(n):
            d["desc"] = ("、".join(d["combo"]) + "一起说。" if isinstance(d["combo"], list)
                         else COLLECTIVE_DESC.get(n, f"{n}一起说，剧本化时按上下文看是谁。"))
        else:
            d["desc"] = DESC_FIX.get(n, dr[n]["desc"].replace("美穗子", "美穰子").replace("过意意不去", "过意不去"))
        if "card" in dr.get(n, {}):
            d["card"] = dr[n]["card"]
        new[n] = d
    return add, new


def write():
    """已有角色：code 下面补 voiceline，pan 下面补 aliases（已有的接着往后加），别的一个字不动；
    新角色按 主要、配角、龙套、合称，各按句数从多到少，接在已有角色后面、音色池前面。"""
    text = CAST.read_text(encoding="utf-8")
    before = yaml.safe_load(text)
    add, new = entries()
    assert text.startswith(HEAD_OLD) or text.startswith(HEAD_NEW)
    text = text.replace(HEAD_OLD, HEAD_NEW, 1)
    out, cur = [], None
    for ln in text.split("\n"):
        if ln.startswith("  ") and not ln.startswith("    ") and ln.rstrip().endswith(":"):
            cur = ln.strip()[:-1]
        d = add.get(cur, {})
        had = "aliases" in before["characters"].get(cur, {})
        if ln.startswith("    aliases:") and "aliases" in d:
            ln = f"    aliases: [{', '.join(d['aliases'])}]"
        out.append(ln)
        if ln.startswith("    code:") and "voiceline" in d:
            out.append(f"    voiceline: {d['voiceline']}")
        if ln.startswith("    pan:") and "aliases" in d and not had:
            out.append(f"    aliases: [{', '.join(d['aliases'])}]")
        if ln.startswith("# 龙套音色池") and new:
            body = []
            order = ("主要", "配角", "龙套", "合称")
            for n in sorted(new, key=lambda n: order.index(new[n]["tier"])):  # sorted 是稳定的，组内保留句数顺序
                body += emit(n, new[n])
            out[-1:] = body + ["", ln]
    text = "\n".join(out)
    after = yaml.safe_load(text)
    for n, c in before["characters"].items():  # 已有的字段一个不动
        for k, v in c.items():
            if k == "aliases":
                assert after["characters"][n][k][:len(v)] == v, n
            else:
                assert after["characters"][n][k] == v, (n, k)
    assert {k: v for k, v in after.items() if k != "characters"} == {k: v for k, v in before.items() if k != "characters"}
    names = set(after["characters"])
    alias = {a for c in after["characters"].values() for a in c.get("aliases", [])}
    miss = {x["raw"] for x in book_lines() if x["raw"] not in UNNAMED and x["raw"] not in names | alias}
    assert not miss, f"剧本里的说话人不在角色表里：{miss}"
    CAST.write_text(text, encoding="utf-8", newline="\n")
    print(CAST, f"已有 {len(before['characters'])}（补了 {len(add)} 个），新加 {len(new)}，共 {len(names)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"count": count, "draft": draft, "write": write}[sys.argv[1]]()
