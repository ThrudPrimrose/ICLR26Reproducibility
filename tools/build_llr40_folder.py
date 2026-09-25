"""Lay the LLR40 final submissions of a regrade pack out per kernel and language beside the corpus
reference, plus an INDEX.md of the most interesting samples. Read-only on the pack and the corpus.

    python tools/build_llr40_folder.py <pack dir> <out dir> [--corpus <loop_level_reasoning dir>]
"""

import argparse
import collections
import json
import os
import pathlib
import re
import shutil
import statistics
from typing import Any

from daint_pack import backend

EXT = {"c": ".c", "c-openmp": ".c", "hip": ".hip", "triton": ".py", "fortran": ".f90"}
REFERENCE_SUFFIXES = (".py", ".c", ".h", ".cpp", ".yaml", ".f90")
#: (speed-up, setup, file name) of one submission.
Entry = tuple[float, str, str]


def lay_out(rows: list[dict[str, Any]], pack: pathlib.Path, out: pathlib.Path) -> dict[tuple[str, str], list[Entry]]:
    """Copy every submission to <out>/<kernel>/<language>/ and return them by (kernel, language)."""
    by: dict[tuple[str, str], list[Entry]] = collections.defaultdict(list)
    for row in rows:
        lang = backend(row)
        setup = re.sub(r"^(cpf|gpu)-llr-focus40-", "", row["arm"])
        tag = "__lastsaved" if row["env"].get("HPCAGENT_BENCH_REGRADE_SOURCE") == "last_saved" else ""
        name = f"{setup}__S{row['speedup']:.2f}__{row['run_id'].split('.')[-1]}{tag}{EXT[lang]}"
        folder = out / row["benchmark"] / lang
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["source"].replace("@PACK@", str(pack)), folder / name)
        if row.get("device_source"):
            shutil.copyfile(row["device_source"].replace("@PACK@", str(pack)),
                            folder / name.replace(EXT[lang], ".device" + EXT[lang]))
        by[(row["benchmark"], lang)].append((row["speedup"], setup, name))
    return by


def copy_references(kernels: list[str], corpus: pathlib.Path, out: pathlib.Path) -> None:
    for kernel in kernels:
        ref = out / kernel / "reference"
        ref.mkdir(parents=True, exist_ok=True)
        src_dir = corpus / kernel
        for f in sorted(src_dir.iterdir()) if src_dir.is_dir() else []:
            if f.is_file() and f.suffix in REFERENCE_SUFFIXES:
                shutil.copyfile(f, ref / f.name)


def index_lines(by: dict[tuple[str, str], list[Entry]], kernels: list[str]) -> list[str]:
    counts = collections.Counter(lang for (_, lang), entries in by.items() for _ in entries)
    lines = [
        "# LLR40 submissions: index", "",
        "Final submission of every episode (" + ", ".join(f"{n} {lang}"
                                                          for lang, n in sorted(counts.items())) + ") from the",
        "beverin judge stores, grouped `<kernel>/<language>/`. File name: "
        "`<model>-<setup>__S<speed-up on MI300A>__<worker>`.",
        "Fortran `oss120b-fortran` files are the last file saved in the workspace (graded source purged), "
        "marked `__lastsaved`.",
        "`<kernel>/reference/` holds the corpus files (NumPy reference, Numba baseline, manifest).", "",
        "Speed-ups are the recorded MI300A live grades (not the final 4x5 regrade); unsolved episodes have no row.", ""
    ]
    table = ["| kernel | lang | setup | S | file |", "|---|---|---|---|---|"]
    allrows = [(s, k, lang, setup, n) for (k, lang), entries in by.items() for s, setup, n in entries]
    lines += ["## Most interesting samples", "", "### Largest speed-ups (check for legitimacy)", "", *table]
    for s, k, lang, setup, n in sorted(allrows, reverse=True)[:15]:
        lines.append(f"| {k} | {lang} | {setup} | {s:.1f} | `{k}/{lang}/{n}` |")
    lines += ["", "### Slowest correct submissions (S < 0.5)", "", *table]
    for s, k, lang, setup, n in sorted(r for r in allrows if 0 < r[0] < 0.5)[:15]:
        lines.append(f"| {k} | {lang} | {setup} | {s:.2f} | `{k}/{lang}/{n}` |")
    lines += [
        "", "### Largest spread between languages (best per language)", "", "| kernel | best per language |",
        "|---|---|"
    ]
    spread = []
    for k in kernels:
        best = {lang: max(entries)[0] for (kk, lang), entries in by.items() if kk == k and entries}
        if len(best) > 1 and min(best.values()) > 0:
            spread.append((max(best.values()) / min(best.values()), k, best))
    for _, k, best in sorted(spread, reverse=True)[:10]:
        lines.append(f"| {k} | " + ", ".join(f"{lang} {s:.1f}" for lang, s in sorted(best.items())) + " |")
    lines += ["", "## Per kernel", "", "| kernel | language | n | median S | best S (setup) |", "|---|---|---|---|---|"]
    for (k, lang), entries in sorted(by.items()):
        top = max(entries)
        median = statistics.median(e[0] for e in entries)
        lines.append(f"| {k} | {lang} | {len(entries)} | {median:.2f} | {top[0]:.1f} ({top[1]}) |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pack", type=pathlib.Path, help="regrade pack holding wl-*.portable.jsonl and files/")
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("--corpus",
                        type=pathlib.Path,
                        help="default: $HPCAGENT_BENCH/hpcagent_bench/benchmarks/"
                        "loop_level_reasoning")
    args = parser.parse_args()
    corpus = args.corpus
    if corpus is None:
        bench = os.environ.get("HPCAGENT_BENCH")
        if not bench:
            parser.error("pass --corpus or set HPCAGENT_BENCH")
        corpus = pathlib.Path(bench) / "hpcagent_bench" / "benchmarks" / "loop_level_reasoning"
    rows = [json.loads(line) for f in sorted(args.pack.glob("wl-*.portable.jsonl")) for line in f.open()]
    if not rows:
        parser.error(f"no wl-*.portable.jsonl rows in {args.pack}")
    by = lay_out(rows, args.pack, args.out)
    kernels = sorted({k for k, _ in by})
    copy_references(kernels, corpus, args.out)
    (args.out / "INDEX.md").write_text("\n".join(index_lines(by, kernels)) + "\n")
    print(len(rows), "files;", len(kernels), "kernels")


if __name__ == "__main__":
    main()
