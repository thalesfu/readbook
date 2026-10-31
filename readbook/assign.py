"""没编号的角色分声线（生产 07 号票）：cast.yaml 里还没有 voiceline 的配角、龙套，从声线库里借一条现成的。

规则见合成规格第四节"角色怎么分到声线"和 ADR 0002：
- 对象：cast.yaml 里声线名里没有自己的配角、龙套（06 号票留下的），加上集体合称（CROWD，剧本化列不出人名时
  叠 3 条龙套声线）；
- 能借的声线：声线库里、声优不配主要角色的。主要角色的声线不借；榛名是千晶的声优演的，也不借；
- 兼配（泛称的台词，录音却是某个有专属编号的声优）：对着上下文核对过，是那个角色本人的写在 SAME_PERSON，
  这几句按本人算（念、数句数、查撞声都算本人）；不是本人的写在 OWN_ACTOR，照原作用那个声优的声线（用户定的）；
- 没报名字的 98、99 号句子，看得出是谁的写在 UNNAMED_IS，算进那个人的出场；
- 性别：按除去本人那几句后剩下的录音，98 女、99 男、OWN_ACTOR 按那个声优；男女都有的取多的、平手按 cast.yaml
  （用户定的）；没有录音的按 cast.yaml；
- 一天：照 daily.py 的切法（castfill.book_lines 的 day），路线不同算不同的天；序章一个剧本算一天；
- 由 Claude 排（rank，只发文字）：按性别、年龄、描述，把同性别能借的声线分成"合适""勉强"，按先后排，不合适的
  不列。挑不上的再问一轮（rank_gaps）：只给它那几天没人用的，问有没有勉强能用的，接在第一轮后面；
- 硬约束（violations，由程序查，不靠 Claude）：这个角色开口的每一天，没有同一个声优配的别的角色开口，同一个
  声优的不同声线也算（20 佐藤、36 高柳都是三戸耕三）。群声之间不查（都是路人）；BORROW 是用户定的例外；
- 怎么挑（solve_all）：句数多的先挑，挑排得最前、和同天已经定了的不撞的那条（OWN_ACTOR 原作那条排最前）；
  挑不到的，让撞了的同天角色连着往后换，最多 DEPTH 层，不换出新的缺口；补完以后排在前面的又空出来了就换回去；
  角色都定了再挑群声：Claude 列的排完接上其余能借的，男女都要有，凑不齐就用剩下的补满；
- 同名全书一条；挑不到的进缺口清单，交给外来声线专项。

    uv run python -m readbook.assign count      # 数：每个角色的句数、天数、性别，能借的声线和被同天排除掉的
    uv run python -m readbook.assign rank       # claude -p 排声线，缓存在 out/assign/，重跑只补没做的
    uv run python -m readbook.assign rank_gaps  # 挑不上的再问一轮，存 out/assign/rank-gap.json
    uv run python -m readbook.assign solve      # 挑、查硬约束，写 out/assign/分配.json，打印缺口
    uv run python -m readbook.assign write      # 写进 cast.yaml 的 voiceline，按 cast.yaml 再查一遍，写分配表 voices/分配.md
    uv run python -m readbook.assign check      # 只按 cast.yaml 查硬约束
"""

import json
import sys
from collections import Counter, defaultdict
from functools import cache

import yaml

from . import castfill as cf
from . import voicelines as vl
from .trial import BOOK, OUT

CAST = BOOK / "cast.yaml"
CACHE = OUT / "assign"
TABLE = BOOK / "voices" / "分配.md"
FIRST = ("01 第二部-共通", ("2010-11-26", "2010-11-27", "2010-12-01", "2010-12-02", "2010-12-03", "2010-12-05",
                          "2010-12-06", "2010-12-08", "2010-12-09", "2010-12-10", "2010-12-11"))  # 第一段的 11 天

# 兼配里其实是本人的：(名字, 录音编号) → 本人。都对着上下文核对过
SAME_PERSON = {
    ("女子学生１", "03"): "小春",     # 2002（11-26）「是北原春希吧？」小春初登场
    ("女子学生１", "24"): "亚子",     # 2003「早安，小春」，小春回「早安，早百合、亚子」
    ("女子学生２", "23"): "早百合",   # 同上
    ("女子学生２", "04"): "千晶",     # 2001，下一句就是「和泉」（千晶）
    ("女子学生２", "19"): "朋",       # 2031，同伴叫她「朋…」
    ("女学生２", "19"): "朋",         # 1008_030「我绝对不会原谅你的小木曾雪菜」
    ("女生２", "19"): "朋",           # 1008，同伴说「比朋更加想要显眼」
    ("女生１", "25"): "美穰子",       # 2001，春希「是矢田啊」
    ("女性", "16"): "曜子",           # 1008 问路的，1011_030 曜子自己说「那个时候真是帮了大忙了」
    ("客人１", "03"): "小春",         # 2015：值早班的两人下班后坐下来挖苦「啰嗦的指导员」，被提醒了 5 次的新人
    ("客人２", "22"): "中川",         # 2015：同上，和小谷同一时期进店的老店员
    ("女子店员１", "22"): "中川",     # 2005，春希「中川小姐…好久不见」
    ("男性团员１", "28"): "吉田",     # 2406 摔门不演的男主角演员，2407 团长「吉田…拜托了啊？」
    ("男性教师", "18"): "诹访老师",   # 1004 抓翻墙的春希；18 号就是诹访老师，第二天 1005 出场
}
# 真兼配：原作用有编号的声优配了别人。照原作用那个声优的声线
OWN_ACTOR = {"大学女生"}  # 3111 孝宏的女朋友，「不知不觉已经过去四年了」，不是亚子（亚子 2011 年才上大学）

# 没报名字的 98、99 号，看得出是谁的：(剧本, 行号) → 名字
UNNAMED_IS = {
    ("2005", 9): "店长",   # 12-05 打来电话「是北原君吗？」，接着说店长扭了腰；店长 12-11 再出场
    ("2029", 696): "ＤＪ",  # 2011-02-10 广播研究会「请进～！」，同一场 ＤＪ 接着说
}

# 集体合称列不出人名时（剧本化退回「路人」），叠几条龙套声线（06 号票，用户定的）。全员、全体这些一般列得出
# 在场的人，也配上，备着；小木曽家的众人是有名有姓的一家人，不配路人的声音
CROWD = set(cf.COLLECTIVE) - {"小木曽家的众人"}
CROWD_N = 3

# 硬约束的例外（用户定的："借同场男生的声线"）：12-11 联谊会男声龙套 11 个，能借的男声优只剩 10 个、年轻的 5 个，
# 这三个两轮都分不到。借同场男生的声线，2009#355–357 连着的三句各用一条不同的，都对得上借给他们的人自己的台词
BORROW = {
    "男子学生４": "男子学生１",  # 「要接着去卡拉ＯＫ唱歌的人～」，男子学生１ 接着就拉雪菜去唱卡拉ＯＫ
    "男子学生５": "男子学生２",  # 「要留在这里喝酒的人～」，男子学生２ 接着说「那就留在这里吧…再喝酒吧？」
    "男子学生６": "干事",        # 「出租车来了哦。快上快上～」，组织的人叫车
}


# ---------- 数 ----------

def day_of(x: dict) -> str:
    d = x["day"]
    if d.startswith("201-"):  # 2029 里把 2011 写成了 201，daily.py 按 2011 处理
        d = "2011" + d[3:]
    return f"{x['route']}/{d}"


@cache
def cast() -> dict:
    return yaml.safe_load(CAST.read_text(encoding="utf-8"))["characters"]


@cache
def lines() -> dict:
    return vl.load()


@cache
def owners() -> dict[str, str]:
    return {c["code"]: n for n, c in cast().items() if c.get("code") and c["code"] not in vl.SHARED}


def code(x: dict) -> str | None:
    return vl.code_of(x["voice"]) if x["voice"] else None


def speakers_of(x: dict) -> list[str]:
    """一句书里的台词，有声书里由谁念（角色表的名字）。合称：点了名的拆成各人，集体的就是合称自己。"""
    if x["raw"] in vl.UNNAMED:
        if (x["sid"], x["n"]) in UNNAMED_IS:
            return [UNNAMED_IS[x["sid"], x["n"]]]
        o = owners().get(code(x))
        return [o] if o else []
    n = cf.main_name(x["raw"])
    if "＆" in n:
        return [cf.main_name(m) for m in n.split("＆")]
    return [SAME_PERSON.get((n, code(x)), n)]


def actor(line: str) -> str:
    v = lines()[line]
    return v["声优"] or f"{v['编号']}号"


@cache
def stats() -> dict:
    """{名字: 句数、录音编号、天数、句子下标}，按 speakers_of 算到人；句数按语音文件去重（同 castfill）。"""
    seen, out = set(), {}
    for i, x in enumerate(cf.book_lines()):
        for n in speakers_of(x):
            s = out.setdefault(n, {"lines": 0, "codes": Counter(), "days": set(), "rows": []})
            s["days"].add(day_of(x))
            s["rows"].append(i)
            if x["voice"] and (x["voice"], n) in seen:
                continue
            seen.add((x["voice"], n))
            s["lines"] += 1
            if x["voice"]:
                s["codes"][code(x)] += 1
    return out


@cache
def presence() -> dict[str, set[str]]:
    out = defaultdict(set)
    for n, s in stats().items():
        for d in s["days"]:
            out[d].add(n)
    return dict(out)


@cache
def targets() -> list[str]:
    """要分的：非合称、全书有出场（按 speakers_of 算到人）、声线名里没有自己的（不是主角色，不是榛名、吉田这种
    自成一线的，不是女主角１～４），加上 CROWD。06 号票留下没有 voiceline 的就是这些；写进 cast.yaml 之后也不变。"""
    return [n for n, c in cast().items() if c["tier"] != "合称" and n in stats()
            and cf.own_line(cf.PLAYED.get(n, n), lines()) is None] + sorted(CROWD)


def check_pairs():
    """要分的角色里，兼配的 (名字, 专属编号) 都得有着落：本人，或者照原作。
    （已有声线的角色录音偶尔是别人的编号，比如春希有 2 句录在 01 号，那是语音包的事，不在这里管。）"""
    bad = Counter()
    for x in cf.book_lines():
        c = code(x)
        if x["raw"] in vl.UNNAMED or not c or c in vl.SHARED:
            continue
        n = cf.main_name(x["raw"])
        if cf.is_combo(n) or cf.own_line(cf.PLAYED.get(n, n), lines()) or (n, c) in SAME_PERSON or n in OWN_ACTOR:
            continue
        bad[n, c] += 1
    assert not bad, f"兼配的句子没有着落：{dict(bad)}"


def voice_gender(n: str) -> tuple[str, bool]:
    """(按录音的性别, 男女都有)。98 女、99 男、OWN_ACTOR 按那个声优；票多的；没有录音的按 cast.yaml。"""
    votes = Counter()
    for c, k in stats()[n]["codes"].items():
        if c == cf.FEMALE_MOB:
            votes["女"] += k
        elif c == cf.MALE_MOB:
            votes["男"] += k
        elif n in OWN_ACTOR:
            votes["男" if c in vl.MALE else "女"] += k
    if not votes:
        return cast()[n]["gender"], False
    top = votes.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return cast()[n]["gender"], True
    return top[0][0], len(top) > 1


@cache
def lendable() -> list[str]:
    main = {actor(c["voiceline"]) for c in cast().values() if c["tier"] == "主要"}
    return [n for n in lines() if actor(n) not in main]


def fixed_line(n: str) -> list[str]:
    v = cast().get(n, {}).get("voiceline")
    return [] if v is None else [v] if isinstance(v, str) else v


def blocked(n: str) -> dict[str, list[str]]:
    """同天开口、本来就有声线的角色（不是这张票分的）占掉的声优 → ["天 角色"]。"""
    out = defaultdict(list)
    for d in sorted(stats()[n]["days"]):
        for o in sorted(presence()[d] - {n} - set(targets())):
            for ln in fixed_line(o):
                out[actor(ln)].append(f"{d} {o}")
    return dict(out)


def first_segment(n: str) -> int:
    route, days = FIRST
    return sum(f"{route}/{d}" in stats()[n]["days"] for d in days)


def count():
    check_pairs()
    st = stats()
    todo = targets()
    left = {n for n, c in cast().items() if "voiceline" not in c and c["tier"] != "合称" and n in st}
    assert left <= set(todo), f"没有声线、又不在要分的里面：{left - set(todo)}"
    print(f"要分的 {len(todo)} 个（群声 {len(CROWD)}）；能借的声线 {len(lendable())} 条：{lendable()}")
    for n in todo:
        g, mixed = voice_gender(n) if n not in CROWD else ("男女", False)
        cand = [ln for ln in lendable() if g == "男女" or lines()[ln]["性别"] == g]
        b = blocked(n)
        free = [ln for ln in cand if actor(ln) not in b]
        flag = ("混" if mixed else "") + ("≠表" if n not in CROWD and g != cast()[n]["gender"] else "")
        print(f"{n}\t{g}{flag}\t{st[n]['lines']} 句\t{len(st[n]['days'])} 天\t第一段 {first_segment(n)} 天\t"
              f"可用 {len(free)}/{len(cand)}")


# ---------- Claude 排声线 ----------

RANK_RULES = """\
你在给《白色相簿2》有声书的配角、龙套挑声线。这些角色在原作里没有自己的声优，要从下面这些现成的声线里借一条来念。
声线是原作某个声优的一种声音，名字写作「声优-这条声线原本配的角色」。时间线：第一部序章是 2007 年峰城大附属高中
三年级；第二部是 2010–2011 年，春希他们上大学；第三部是 2012–2013 年，已经工作。

每条声线给了：性别、原声音高中位（Hz，只作参考，音高分不开男女，也不全代表年龄）、原声句数、这条声线原本配的
角色的描述（里面有声音的描述）。
每个角色给了：按原作录音定的性别、大概年龄、描述、全书句数和出场天数、几句台词。

给每个角色，从同性别的声线里挑出能用的，按合适程度从好到差排：
- 合适：年龄感、音色、气质对得上；
- 勉强：能念，但年龄感或气质差一点；
- 明显不合适的不要列，比如中年大叔的声音配高中男生、温柔少女的声音配粗嗓门的大妈；
- 每条写一句理由，20 字以内，说哪里对得上或者差在哪；
- 不用管同一天会不会和别人撞声，程序会查，挑不上的会往后取。所以合适的、勉强的都尽量列全，排好先后；
- 广播、手机提示音这类：要清楚、平稳、像播音的；
- 原声句数少（几十句以下）或者注明「锚点没过门槛」的声线，合成出来更容易不像，同样合适时往后排。
标「群声」的是几个人一起喊（比如店员们一起说早上好），从男女两边都列，程序会挑 3 条叠在一起。

按 schema 返回 characters，每个角色一条，name 照抄，line 照抄声线名。
"""

BATCH = 20


def rank_schema() -> dict:
    return {
        "type": "object",
        "properties": {"characters": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "picks": {"type": "array", "items": {
                "type": "object",
                "properties": {"line": {"type": "string", "enum": lendable()},
                               "fit": {"type": "string", "enum": ["合适", "勉强"]},
                               "why": {"type": "string"}},
                "required": ["line", "fit", "why"]}}},
            "required": ["name", "picks"]}}},
        "required": ["characters"],
    }


def catalog() -> str:
    out = []
    for g in ("男", "女"):
        out.append(f"### {g}声")
        for ln in lendable():
            v = lines()[ln]
            if v["性别"] != g:
                continue
            roles = ln.split("-", 1)[1].split("，")  # 只要声线名里的角色，不要借用这条声线的龙套
            weak = "，锚点没过门槛" if v["锚点像原声"] < v["音色门槛"] else ""
            out.append(f"- {ln}：音高中位 {v['音高中位']:.0f} Hz，原声 {v['原声句数']} 句{weak}。"
                       + " ".join(f"{r}：{cast()[r]['desc']}" for r in roles))
    return "\n".join(out)


def samples(n: str, k: int = 4) -> list[str]:
    rows = cf.book_lines()
    idx = stats()[n]["rows"]
    pick = [idx[round(i * (len(idx) - 1) / max(k - 1, 1))] for i in range(min(k, len(idx)))]
    return [f"  - {day_of(rows[i])}：{rows[i]['text'][:60]}" for i in dict.fromkeys(pick)]


def profile(n: str) -> str:
    s = stats()[n]
    if n in CROWD:
        head = f"## {n}（群声，男女都列）\n- {cast()[n]['desc']}"
    else:
        c = cast()[n]
        g, mixed = voice_gender(n)
        note = ""
        if g != c["gender"]:
            note = f"（角色表写的是{c['gender']}，但除去本人那几句，剩下的录音是{g}声，按{g}声挑；描述按台词看）"
        elif mixed:
            note = "（各处的泛称，录音男女都有，按多的挑）"
        head = f"## {n}\n- 性别：{g}{note}；年龄约 {c['age']}；{c['tier']}，{s['lines']} 句，{len(s['days'])} 天\n- {c['desc']}"
    return head + "\n- 台词：\n" + "\n".join(samples(n))


GAP = CACHE / "rank-gap.json"
GAP_RULES = """\
# 第二轮：挑不上的
下面这些角色，你第一轮排的声线都和同一天开口的别的角色撞了声优（同一天不能有两个角色用同一个声优）。
每个角色后面列了它开口的那几天都没人用、第一轮你没列的声线。看看有没有勉强能用的：
- 能用的按先后排，一般写「勉强」，理由 20 字以内，说差在哪、为什么还能用；
- 都不合适就返回空的 picks，这个角色会进缺口清单，以后从别的作品找声音；
- 只能从这个角色后面列出的声线里挑。
"""


def ranks() -> dict[str, list[dict]]:
    """第一轮排的，后面接上第二轮（缺口再问的，标 round: 2）。"""
    out = {}
    for p in sorted(CACHE.glob("rank-[0-9]*.json")):
        out |= {c["name"]: c["picks"] for c in json.loads(p.read_text(encoding="utf-8"))}
    if GAP.exists():
        for n, picks in json.loads(GAP.read_text(encoding="utf-8")).items():
            out[n] = out.get(n, []) + [{**p, "round": 2} for p in picks]
    return out


def spare(n: str, got: dict[str, list[str]], rk: dict) -> list[str]:
    """n 开口的那几天都没人用、又不在它已经排过的里面的同性别声线（群声的分配不算，它们最后才挑）。"""
    nb = neighbors(targets())[n]
    used = {actor(x) for o in nb if o not in CROWD for x in got.get(o, [])} | set(blocked(n))
    g = voice_gender(n)[0]
    return [ln for ln in lendable() if lines()[ln]["性别"] == g and actor(ln) not in used and ln not in options(n, rk)]


def rank_gaps():
    """缺口再问一轮：只给它那几天都没人用的声线，问有没有勉强能用的。结果接在第一轮后面，存 rank-gap.json。"""
    from .scriptize import ask_claude

    rk = ranks()
    got = solve_all(rk)
    have = json.loads(GAP.read_text(encoding="utf-8")) if GAP.exists() else {}
    todo = {n: spare(n, got, rk) for n in targets() if n not in CROWD and not got[n] and n not in have}
    if not todo:
        print("没有要再问的")
        return
    for n, s in todo.items():
        if not s:
            have[n] = []
    ask = {n: s for n, s in todo.items() if s}
    if ask:
        body = "\n\n".join(profile(n) + "\n- 那几天没人用的声线：" + "、".join(s) for n, s in ask.items())
        meta = {}
        data = ask_claude(RANK_RULES + "\n# 能借的声线\n" + catalog() + "\n\n" + GAP_RULES + "\n" + body,
                          schema=rank_schema(), meta=meta)
        got2 = {c["name"]: [p for p in c["picks"] if p["line"] in ask.get(c["name"], [])] for c in data["characters"]}
        have |= {n: got2.get(n, []) for n in ask}
        print(f"第二轮 {len(ask)} 个，{meta.get('duration_ms', 0) / 1000:.0f} 秒，${meta.get('total_cost_usd', 0):.2f}")
    GAP.write_text(json.dumps(have, ensure_ascii=False, indent=1), encoding="utf-8")
    for n in todo:
        print(n, "那几天没人用的：", todo[n], "→", [(p["line"], p["fit"], p["why"]) for p in have[n]])


def rank():
    """claude -p 按批排声线（只发文字），缓存在 out/assign/rank-XX.json，重跑只补没做的。"""
    from concurrent.futures import ThreadPoolExecutor

    from .scriptize import ask_claude

    CACHE.mkdir(parents=True, exist_ok=True)
    have = ranks()
    todo = [n for n in targets() if n not in have]
    first = len(list(CACHE.glob("rank-[0-9]*.json")))
    head = RANK_RULES + "\n# 能借的声线\n" + catalog() + "\n\n# 要挑的角色\n"
    jobs = [(CACHE / f"rank-{first + b // BATCH:02d}.json", todo[b:b + BATCH]) for b in range(0, len(todo), BATCH)]

    def run(job):
        path, names = job
        meta = {}
        data = ask_claude(head + "\n\n".join(profile(n) for n in names), schema=rank_schema(), meta=meta)
        got = {c["name"]: c for c in data["characters"]}
        missing = [n for n in names if n not in got]
        if missing:
            raise ValueError(f"{path.name} 漏了 {missing}")
        path.write_text(json.dumps([got[n] for n in names], ensure_ascii=False, indent=1), encoding="utf-8")
        print(path.name, f"{meta.get('duration_ms', 0) / 1000:.0f} 秒", f"${meta.get('total_cost_usd', 0):.2f}", flush=True)

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(run, jobs))


# ---------- 挑、查 ----------

def options(n: str, rk: dict) -> list[str]:
    """按 Claude 排的先后：合适的在前、勉强的在后，性别不对的去掉；OWN_ACTOR 原作那条排最前。"""
    picks = [p for p in rk.get(n, []) if p["line"] in lendable()
             and (n in CROWD or lines()[p["line"]]["性别"] == voice_gender(n)[0])]
    order = [p["line"] for p in picks if p["fit"] == "合适"] + [p["line"] for p in picks if p["fit"] == "勉强"]
    if n in OWN_ACTOR:
        order = [vl.line_of(c, n) for c in stats()[n]["codes"]] + order
    if n in CROWD:  # 群声是几个路人一起喊，Claude 列的排完，接上其余能借的
        order += [ln for ln in lendable() if "女" not in n or lines()[ln]["性别"] == "女"]
    return list(dict.fromkeys(order))


def neighbors(names: list[str]) -> dict[str, set[str]]:
    """要分的角色里，同一天开口的。"""
    out = {n: set() for n in names}
    for ss in presence().values():
        here = [s for s in ss if s in out]
        for s in here:
            out[s] |= set(here) - {s}
    return out


def crowd_ok(n: str, picked: list[str], ln: str) -> bool:
    """群声：名字里有「女」的全用女声；别的男女都要有，同一个性别最多 CROWD_N - 1 条。"""
    g = lines()[ln]["性别"]
    if "女" in n:
        return g == "女"
    return sum(lines()[p]["性别"] == g for p in picked) < CROWD_N - 1


DEPTH = 3  # 补缺口时最多连着换几层


def solve_all(rk: dict) -> dict[str, list[str]]:
    """句数多的先挑：按 options 的先后，挑第一条不和同天已有声线的角色撞声优的。
    挑不到的补缺口：拿它能用的一条，撞了的同天角色换到它自己能用的另一条，那条再撞就接着换，最多 DEPTH 层，
    换不开就不动（不会换出新的缺口）。角色都定了再挑群声。"""
    names, st = targets(), stats()
    nb, blk = neighbors(names), {n: set(blocked(n)) for n in names}
    order = sorted((n for n in names if n not in CROWD), key=lambda n: (-st[n]["lines"], -len(st[n]["days"]), n))
    got: dict[str, list[str]] = {}

    def free(n, ln):  # 群声之间不查：都是不报名字的路人，同一天的「女店员们」「女子店员们」就是同一群人
        a = actor(ln)
        return a not in blk[n] and all(a != actor(x) for o in nb[n] if not (n in CROWD and o in CROWD)
                                       for x in got.get(o, []))

    def place(g, n, ln, depth, frozen):
        """把 ln 给 n，撞了的同天角色挪开；成了返回新的分配，不成返回 None。"""
        a = actor(ln)
        if a in blk[n]:
            return None
        g = {**g, n: [ln]}
        for h in sorted(nb[n]):
            if not any(actor(x) == a for x in g.get(h, [])):
                continue
            if h in frozen or depth == 0:
                return None
            for alt in options(h, rk):
                if actor(alt) != a and alt not in got[h]:
                    r = place(g, h, alt, depth - 1, frozen | {n})
                    if r is not None:
                        g = r
                        break
            else:
                return None
        return g

    for n in order:
        got[n] = next(([ln] for ln in options(n, rk) if free(n, ln)), [])
    for n in order:  # 补缺口
        if got[n]:
            continue
        for ln in options(n, rk):
            r = place(got, n, ln, DEPTH, frozen=set())
            if r is not None:
                moved = [f"{o} {got[o][0].split('-', 1)[1]}→{r[o][0].split('-', 1)[1]}" for o in order
                         if o != n and r[o] != got[o]]
                print(f"补缺口：{n} 用 {ln}；{'，'.join(moved)}", flush=True)
                got = r
                break
    while True:  # 补缺口挪过的，排在前面的那条后来又空出来了，就换回去
        back = 0
        for n in order:
            opts = options(n, rk)
            if got[n] and got[n][0] in opts:
                better = next((ln for ln in opts[:opts.index(got[n][0])] if free(n, ln)), None)
                if better:
                    print(f"换回：{n} {got[n][0]} → {better}", flush=True)
                    got[n], back = [better], back + 1
        if not back:
            break
    for n, lender in BORROW.items():  # 用户定的例外：还是缺口的，借同场那个人的声线
        if not got[n] and got[lender]:
            got[n] = list(got[lender])
    for n in sorted(CROWD):
        picked = []
        for balance in (True, False):  # 男女凑不齐就不管搭配，用剩下能用的补满
            for ln in options(n, rk):
                if len(picked) < CROWD_N and free(n, ln) and actor(ln) not in {actor(p) for p in picked} \
                        and (crowd_ok(n, picked, ln) if balance else "女" not in n or lines()[ln]["性别"] == "女"):
                    picked.append(ln)
        got[n] = picked
    return got


def violations(assign: dict[str, list[str]]) -> list[str]:
    """硬约束：一天里同一个声优（不分声线）只配一个角色，至少有一个是这张票分的才算；群声之间不算（都是路人），
    BORROW 和借给它的人不算（用户定的例外）；群声自己的几条也不能同一个声优。"""
    def of(s):
        return assign[s] if s in assign else fixed_line(s)
    out = []
    for n, ls in assign.items():
        if len({actor(x) for x in ls}) < len(ls):
            out.append(f"{n} 的几条声线有同一个声优：{ls}")
    for d, ss in sorted(presence().items()):
        by = defaultdict(set)
        for s in ss:
            for x in of(s):
                by[actor(x)].add(s)
        for a, who in sorted(by.items()):
            who = {s for s in who if BORROW.get(s) not in who}
            if len(who) > 1 and who & set(assign) and not who <= CROWD:
                out.append(f"{d}：{a} 配了 {'、'.join(sorted(who))}")
    return out


def solve():
    rk = ranks()
    miss = [n for n in targets() if n not in rk]
    assert not miss, f"还没排：{miss}"
    got = solve_all(rk)
    bad = violations({n: v for n, v in got.items() if v})
    assert not bad, bad
    (CACHE / "分配.json").write_text(json.dumps(got, ensure_ascii=False, indent=1), encoding="utf-8")
    gaps = [n for n, v in got.items() if len(v) < (CROWD_N if n in CROWD else 1)]
    first = Counter(options(n, rk).index(v[0]) + 1 if v and v[0] in options(n, rk) else 0 for n, v in got.items()
                    if n not in CROWD)
    print(f"分了 {sum(bool(v) for v in got.values())}/{len(got)}，排第几的：{dict(sorted(first.items()))}，"
          f"缺口 {len(gaps)}：{gaps}")


# ---------- 写 ----------

VOICELINE_OLD = "#       照声线库填；千晶演的女主角１～４用千晶的（合成规格第四节）；其余留给 07 号票分。\n"
VOICELINE_NEW = ("#       照声线库填；千晶演的女主角１～４用千晶的（合成规格第四节）。没编号的配角、龙套，07 号票从声线库里借\n"
                 "#       （readbook/assign.py，分配表 voices/分配.md）；列不出人名的集体合称写 3 条，叠在一起；\n"
                 "#       分不到的不写，在分配表的缺口清单里，等外来声线。\n")


def from_cast() -> dict[str, list[str]]:
    return {n: fixed_line(n) for n in targets() if fixed_line(n)}


def check():
    bad = violations(from_cast())
    print(f"按 cast.yaml 查：{len(from_cast())} 个有声线，违反 {len(bad)} 处", *bad, sep="\n")
    return bad


def write():
    got = json.loads((CACHE / "分配.json").read_text(encoding="utf-8"))
    text = CAST.read_text(encoding="utf-8")
    before = yaml.safe_load(text)
    assert set(got) == set(targets()), "分配和要分的对不上，重跑 solve"
    text = text.replace(VOICELINE_OLD, VOICELINE_NEW, 1)
    def line(n):
        return f"    voiceline: {got[n][0]}" if n not in CROWD else f"    voiceline: [{', '.join(got[n])}]"

    out, cur = [], None
    for ln in text.split("\n"):
        if ln.startswith("  ") and not ln.startswith("    ") and ln.rstrip().endswith(":"):
            cur = ln.strip()[:-1]
        c = before["characters"].get(cur, {})
        if cur in got and ln.startswith("    voiceline:"):  # 这张票以前写的：按这次的分配改，分不到了就去掉
            if got[cur]:
                out.append(line(cur))
            continue
        out.append(ln)
        if cur in got and got[cur] and "voiceline" not in c and ln.startswith("    code:" if "code" in c else "    tier:"):
            out.append(line(cur))  # 照 castfill.FIELDS 的顺序，放在 code（没有就 tier）后面
    text = "\n".join(out)
    after = yaml.safe_load(text)
    for n, c in before["characters"].items():  # 只动这张票分的 voiceline，别的一个字不动
        keep = {k: v for k, v in c.items() if not (n in got and k == "voiceline")}
        assert {k: v for k, v in after["characters"][n].items() if not (n in got and k == "voiceline")} == keep, n
    assert {k: v for k, v in after.items() if k != "characters"} == {k: v for k, v in before.items() if k != "characters"}
    CAST.write_text(text, encoding="utf-8", newline="\n")
    cast.cache_clear()
    bad = check()
    assert not bad
    table(got)


def table(got: dict[str, list[str]]):
    """分配表：每个角色的出场天数、句数、声线和理由，缺口清单，兼配按本人算的，每条声线借了几次。"""
    rk, st, names = ranks(), stats(), targets()
    nb, blk = neighbors(names), {n: blocked(n) for n in names}
    why = {n: {p["line"]: p for p in rk[n]} for n in names}

    def skipped(n, ln):
        a = actor(ln)
        if a in blk[n]:
            day, who = blk[n][a][0].rsplit(" ", 1)
            return f"{day.split('/', 1)[1]} {who}开口"
        o = next((o for o in sorted(nb[n]) if o not in BORROW and (n in CROWD or o not in CROWD)  # 群声最后才挑
                  and any(actor(x) == a for x in got.get(o, []))), None)
        if o is None:
            return "？"
        return f"同天的{o}用了{'' if ln in got[o] else '同一个声优的'}"

    def reason(n, ln):
        opts = options(n, rk)
        if n in OWN_ACTOR and ln == opts[0]:
            return "原作就是这个声优配的（用户定的）"
        if n in BORROW:
            return (f"**例外**：能用的都和同天的人撞了，借同场{BORROW[n]}的（用户定的）；"
                    + "、".join(f"{x.split('-', 1)[1]}（{skipped(n, x)}）" for x in opts))
        p = why[n].get(ln, {})
        k = opts.index(ln)
        r = f"{'第二轮，' if p.get('round') == 2 else ''}{p.get('fit', '')}：{p.get('why', '')}"
        if k:
            r += "；排在前面的" + "、".join(f"{x.split('-', 1)[1]}（{skipped(n, x)}）" for x in opts[:k])
        return r

    def short(ln):
        return ln.split("-", 1)[1] + "的"

    route, days = FIRST
    seg = {n for n in names if first_segment(n)}
    gaps = [n for n in names if len(got[n]) < (CROWD_N if n in CROWD else 1)]
    used = Counter(x for v in got.values() for x in v)
    rows = sorted((n for n in names if n not in CROWD), key=lambda n: (-st[n]["lines"], -len(st[n]["days"]), n))
    md = ["# 没编号的角色分声线（生产 07 号票）", "",
          "由 `uv run python -m readbook.assign write` 生成，规则见 `readbook/assign.py` 和合成规格第四节。", "",
          f"- **分了** {len(names) - len(gaps)}/{len(names)}：角色 {len(rows)}、列不出人名的集体合称 {len(CROWD)}（每个 3 条）；"
          f"**缺口** {len(gaps)}；**例外** {len(BORROW)}（{'、'.join(BORROW)}，借同场男生的，用户定的）；",
          "- **硬约束**：每个角色开口的每一天，没有同一个声优配的别的角色开口，同一个声优的不同声线也算"
          "（`assign.check` 按 cast.yaml 查过，0 处；群声之间、上面的例外不查）；",
          f"- **能借的声线** {len(lendable())} 条：主要角色的声优不借（榛名是千晶的声优，也不借）；",
          "- **怎么挑**：Claude 按性别、年龄、描述把能借的声线分成合适、勉强，按先后排；程序按句数从多到少，"
          "挑排得最前、同天不撞声的那条；",
          f"- **第一段**（{route} {days[0][5:]}～{days[-1][5:]}）出场的标了 ★。", "",
          "## 分配", "",
          "| 角色 | 性别 | 句数 | 天数 | 声线 | 理由 |", "|---|---|---|---|---|---|"]
    for n in rows:
        g, mixed = voice_gender(n)
        gn = g + ("（混）" if mixed else "") + (f"（表上{cast()[n]['gender']}）" if g != cast()[n]["gender"] else "")
        v = got[n]
        md.append(f"| {'★' if n in seg else ''}{n} | {gn} | {st[n]['lines']} | {len(st[n]['days'])} | "
                  f"{v[0] if v else '**缺口**'} | {reason(n, v[0]) if v else ''} |")
    md += ["", "## 群声（列不出人名的集体合称，叠 3 条）", "", "| 合称 | 天数 | 声线 | 理由 |", "|---|---|---|---|"]
    for n in sorted(CROWD):
        md.append(f"| {'★' if n in seg else ''}{n} | {len(st[n]['days'])} | {'、'.join(got[n]) or '**缺口**'} | "
                  + "；".join(f"{short(x)}{why[n][x]['why']}" if x in why[n] else f"{short(x)[:-1]}（Claude 没列，凑数补的）"
                             for x in got[n]) + " |")
    md += ["", "## 缺口清单（交给外来声线专项）", ""]
    for n in gaps:
        opts = options(n, rk)
        md.append(f"- **{n}**（{st[n]['lines']} 句，{len(st[n]['days'])} 天）：能用的 {len(opts)} 条都撞了——"
                  + "；".join(f"{short(x)}{skipped(n, x)}" for x in opts))
    if not gaps:
        md.append("没有。")
    md += ["", "硬约束的例外（用户定的“借同场男生的声线”，外来声线到了可以换掉）：", ""]
    md += [f"- **{n}**（{st[n]['lines']} 句，{'、'.join(d.split('/', 1)[1] for d in sorted(st[n]['days']))}）："
           f"借{lender}的 {short(got[n][0])[:-1]}" for n, lender in BORROW.items()]
    md += ["", "## 兼配：按本人算的", "",
           "泛称的台词、录音却是有编号的声优的，对着上下文核对过是本人，这几句按本人念（`assign.SAME_PERSON`）：", ""]
    md += [f"- {n}（{c} 号）→ **{who}**" for (n, c), who in SAME_PERSON.items()]
    md += ["", "全是本人、名字底下不剩句子的，不分声线：" + "、".join(
        sorted({n for n, _ in SAME_PERSON} - set(st))) + "。", "",
        "照原作用那个声优的（`assign.OWN_ACTOR`）：" + "、".join(sorted(OWN_ACTOR)) + "。", "",
        "## 每条声线借了几次", ""]
    md += [f"- {ln}：{used[ln]}" for ln in lendable()]
    TABLE.write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print(TABLE)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"count": count, "rank": rank, "rank_gaps": rank_gaps, "solve": solve, "write": write,
     "check": check}[sys.argv[1]]()
