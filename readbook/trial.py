"""试做驱动：A＋B 两段的剧本化与盲测。

    uv run python -m readbook.trial scriptize   # 写 books/wa2/script/*.jsonl 和 out/blind/*.jsonl
    uv run python -m readbook.trial evaluate    # 盲测准确率
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from . import wa2
from .scriptize import load_cast, scriptize

ROOT = Path(__file__).resolve().parent.parent
BOOK = ROOT / "books" / "wa2"
OUT = ROOT / "out"

# 试做片段：(文件名, 路线, 日期)
SEGMENTS = [
    ("02-2011-01-01", "02 第二部-雪菜线", (2011, 1, 1)),
    ("07-2013-01-26", "07 第三部-共通前半", (2013, 1, 26)),
]


def raw_lines(route, date) -> list[dict]:
    rows = []
    for line in wa2.load_day(route, date):
        d = asdict(line)
        d["voice_code"] = line.voice_code
        rows.append(d)
    return rows


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def cmd_scriptize():
    cast = load_cast(BOOK / "cast.yaml")
    jobs = []
    for name, route, date in SEGMENTS:
        lines = raw_lines(route, date)
        write_jsonl(OUT / "raw" / f"{name}.jsonl", lines)
        jobs.append((BOOK / "script" / f"{name}.jsonl", lines, False))
        jobs.append((OUT / "blind" / f"{name}.jsonl", lines, True))

    def run(job):
        path, lines, blind = job
        write_jsonl(path, scriptize(lines, cast, blind=blind))
        return path

    with ThreadPoolExecutor(len(jobs)) as ex:
        for p in ex.map(run, jobs):
            print("写入", p.relative_to(ROOT))


def cmd_evaluate(blind_dir: str = "blind"):
    """盲测：只在原文有名字的台词上比说话人（？？？ 以语音编号还原后的结果为准）。"""
    total = hit = 0
    for name, _, _ in SEGMENTS:
        raw = {r["idx"]: r for r in read_jsonl(OUT / "raw" / f"{name}.jsonl")}
        truth = {r["idx"]: r for r in read_jsonl(BOOK / "script" / f"{name}.jsonl")}
        blind = read_jsonl(OUT / blind_dir / f"{name}.jsonl")
        t = h = 0
        for b in blind:
            if not raw[b["idx"]]["speaker"]:
                continue
            t += 1
            if b["speaker"] == truth[b["idx"]]["speaker"]:
                h += 1
            else:
                print(f"  {name} #{b['idx']} 应为 {truth[b['idx']]['speaker']}，盲测给 {b['speaker']}：{b['text'][:30]}")
        print(f"{name}：{h}/{t} = {h / t:.1%}")
        total += t
        hit += h
    print(f"合计：{hit}/{total} = {hit / total:.1%}")


VERSIONS = {  # 版本 → 混音时是否做内心后期
    "post": True,
    "instruct": False,
    "orig": True,
}
TITLES = {"02-2011-01-01": "2011-01-01 周六（雪菜线·新年）",
          "07-2013-01-26": "2013-01-26 周六（第三部·共通）"}
COVER = Path("~/Library/Mobile Documents/com~apple~CloudDocs/Calibre 书库/九户史明/"
             "白色相簿2 剧本全集 (2528)/cover.jpg").expanduser()


def export_orig(rows: list[dict], out: Path):
    """把这段台词各自的日语原声导出成 48kHz wav。"""
    import soundfile as sf

    from .voices import SR, VoicePak

    vp = None
    out.mkdir(parents=True, exist_ok=True)
    for r in rows:
        if r.get("voice"):
            f = out / r["voice"].replace(".OGG", ".wav")
            if not f.exists():
                vp = vp or VoicePak()
                sf.write(f, vp.audio(r["voice"]), SR)


def cmd_synth(versions: list[str]):
    from .synth import Synth, load_yaml

    cast = load_cast(BOOK / "cast.yaml")
    refs = load_yaml(OUT / "refs" / "refs.yaml")
    synth = Synth(cast, refs, load_yaml(BOOK / "lexicon.yaml"), ROOT)
    segs = {name: read_jsonl(BOOK / "script" / f"{name}.jsonl") for name, _, _ in SEGMENTS}
    synth.design_pool({r["speaker"] for rows in segs.values() for r in rows})
    report = []
    for version in versions:
        synth.stats = {"audio": 0.0, "wall": 0.0}
        for name, rows in segs.items():
            orig = OUT / "orig" / name
            if version == "orig":
                export_orig(rows, orig)
            synth.run(rows, version, OUT / "wav" / version / name, orig,
                      share=OUT / "wav" / "post" / name)
        s = synth.stats
        if s["audio"]:
            report.append(f"{version}：合成 {s['audio']:.0f} 秒音频，用时 {s['wall']:.0f} 秒，实时率 {s['wall'] / s['audio']:.2f}")
    if synth.fallbacks:
        report.append(f"失控后退回 post 版的句子：{'、'.join(synth.fallbacks)}")
    print("\n".join(report))


def cmd_prune():
    """删掉失控（超时长）和只有标点的句子，下次 synth 只重做这些。"""
    import soundfile as sf

    from .synth import SPOKEN, load_yaml, max_seconds, tts_text

    lex = load_yaml(BOOK / "lexicon.yaml")
    for version in VERSIONS:
        n = 0
        for name, _, _ in SEGMENTS:
            for r in read_jsonl(BOOK / "script" / f"{name}.jsonl"):
                f = OUT / "wav" / version / name / f"{r['idx']:04d}.wav"
                text = tts_text(r, lex)
                if f.exists() and (not SPOKEN.search(text) or sf.info(f).duration > max_seconds(text)):
                    f.unlink()
                    n += 1
        print(f"{version}：删掉 {n} 句")


def cmd_mix():
    from .mix import assemble, to_mp3

    cast = load_cast(BOOK / "cast.yaml")
    for version, inner in VERSIONS.items():
        for track, (name, _, _) in enumerate(SEGMENTS, 1):
            wav = OUT / "wav" / version / name
            if not wav.exists():
                continue
            rows = read_jsonl(BOOK / "script" / f"{name}.jsonl")
            audio = assemble(rows, wav, cast, inner)
            mp3 = OUT / "mp3" / version / f"{track:02d} {TITLES[name]}.mp3"
            to_mp3(audio, mp3, {"album": f"白色相簿2 剧本全集（试做·{version}）",
                                "title": TITLES[name], "track": str(track),
                                "artist": "AI 有声书试做"}, COVER)
            print(f"{mp3.relative_to(ROOT)}  {len(audio) / 48000 / 60:.1f} 分钟")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "synth":
        cmd_synth(sys.argv[2:] or list(VERSIONS))
    elif cmd == "evaluate":
        cmd_evaluate(*sys.argv[2:])
    else:
        {"scriptize": cmd_scriptize, "mix": cmd_mix, "prune": cmd_prune}[cmd]()
