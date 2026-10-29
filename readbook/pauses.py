"""18 号票：隐去对白的地方留出停顿。

〈2002〉开头是雪菜回忆里的一段段电话，只写她这头，春希那头的话隐去了。剧本化之后单独问一遍（scriptize.pauses），
只标这一项：这句前面隐去了别人的话没有、要留几秒，其他字段不动；混音时 mix.gap 照 pause 留空。
不重新合成，拿 16 号票全员版的成品句子重新混音，小样也混一遍，看有没有在不该停的地方停。

    uv run python -m readbook.pauses 标注   # 两份剧本各问一遍，写到 books/wa2/trials/pauses/<剧本>-停顿.json（带理由）
    uv run python -m readbook.pauses 混音   # 小样和〈2002〉重新混音：MP3、对照剧本、停顿表.md
    uv run python -m readbook.pauses 写回   # 试听认可后，把 pause 写进 books/wa2/script/*.jsonl
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

from .index_passage import WORK
from .mix import gap
from .passage import remix
from .scriptize import load_cast, pauses, with_pauses
from .trial import BOOK, read_jsonl

TRIAL = BOOK / "trials" / "pauses"
SOURCE = "2.5-原声平均锚点全员"  # 16 号票定下的版本，成品句子在 out/index/<版本>/<片段>/wav/
SEGMENTS = {"小样": "02-2011-01-01", "2010-11-25 〈2002〉": "2010-11-25 〈2002〉"}  # 片段 → 所在剧本


def marks_file(script: str):
    return TRIAL / f"{script}-停顿.json"


def load_marks(script: str) -> dict[int, dict]:
    return {int(k): v for k, v in json.loads(marks_file(script).read_text(encoding="utf-8")).items()}


def cmd_mark():
    cast = load_cast(BOOK / "cast.yaml")
    TRIAL.mkdir(parents=True, exist_ok=True)

    def run(script):
        marks = pauses(read_jsonl(BOOK / "script" / f"{script}.jsonl"), cast)
        marks_file(script).write_text(json.dumps(dict(sorted(marks.items())), ensure_ascii=False, indent=1) + "\n",
                                      encoding="utf-8", newline="\n")
        return script, marks

    with ThreadPoolExecutor(len(SEGMENTS)) as ex:
        for script, marks in ex.map(run, sorted(set(SEGMENTS.values()))):
            print(script, {i: m["pause"] for i, m in marks.items()}, flush=True)


def cmd_mix():
    from .passage import load_rows, stamp

    lines = ["# 18 号票：隐去对白处的停顿", "",
             f"成品句子是 16 号票全员版（`out/index/{SOURCE}/<片段>/wav/`），没有重新合成，只是照剧本的 pause 重新混音。",
             "停顿由 `scriptize.pauses` 标注：这句前面隐去了别人的话没有、要留几秒。时间是空白从哪到哪"
             "（前一句的混响尾巴算在空白里）。原来的间隔：同一人连着说 0.25 秒，换人 0.4 秒，旁白前后 0.5 秒。", ""]
    for name, script in SEGMENTS.items():
        marks = load_marks(script)
        rows = load_rows(name, None, None, None)
        label = f"{name}-IndexTTS-{SOURCE}-停顿"
        starts = remix(name, WORK / SOURCE / name / "wav", TRIAL, label, {i: m["pause"] for i, m in marks.items()})
        mixed = [{**r, "pause": marks[r["idx"]]["pause"] if r["idx"] in marks else 0} for r in rows]
        added = sum(gap(p, r) - gap(p, {**r, "pause": 0}) for p, r in zip(mixed, mixed[1:]))
        lines += [f"## {name}", "", f"MP3：`{label}.mp3`，对照剧本：`{label} 对照剧本.md`。"
                  f"加了停顿的有 {sum(1 for r in mixed[1:] if r['pause'])} 处，总共加长 {added:.1f} 秒。", ""]
        table, same = ["| 空白 | 长 | 前一句 | 这一句 | 为什么 |", "|---|---|---|---|---|"], []
        for i in range(1, len(mixed)):
            p, r = mixed[i - 1], mixed[i]
            if r["pause"]:
                g = gap(p, r)
                table.append(f"| {stamp(starts[i] - g)} → {stamp(starts[i])} | {g:g} 秒 "
                             f"| {p['idx']} {p['speaker']}「{short(p['text'])}」 | {r['idx']} {r['speaker']}「{short(r['text'])}」 "
                             f"| {marks[r['idx']]['why']} |")
            elif r["type"] == p["type"] == "台词" and r["speaker"] == p["speaker"]:
                same.append(f"{p['idx']}→{r['idx']}")
        lines += (table if len(table) > 2 else ["没有标出隐去对白的地方，和原来的混音一样。"]) + [""]
        if same:
            lines += [f"同一人连着说、判断为自己接着说、没留白的：{'、'.join(same)}。", ""]
        rest = {i: m for i, m in marks.items() if i not in {r["idx"] for r in rows}}
        if rest:  # 小样只取了剧本的一部分，剧本其余部分标出的停顿也列出来，看判断得对不对
            lines += [f"剧本 `{script}` 里不在这个片段中的停顿：", ""]
            lines += [f"- {i}（{m['pause']:g} 秒）：{m['why']}" for i, m in sorted(rest.items())]
            lines.append("")
    md = TRIAL / "停顿表.md"
    md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(md)


def short(text: str, n: int = 18) -> str:
    text = text.strip("“”").replace("\n", " ").replace("|", "｜")
    return text if len(text) <= n else text[:n] + "…"


def cmd_write():
    for script in sorted(set(SEGMENTS.values())):
        path = BOOK / "script" / f"{script}.jsonl"
        rows = with_pauses(read_jsonl(path), load_marks(script))
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
        print(path, sum(1 for r in rows if r["pause"]), "处停顿")


if __name__ == "__main__":
    {"标注": cmd_mark, "混音": cmd_mix, "写回": cmd_write}[sys.argv[1]]()
