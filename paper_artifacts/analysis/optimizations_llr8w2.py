"""What the agents actually WROTE, per arm: the optimisation constructs in each accepted answer.

The speed-up tables say how much an arm won; they do not say what it did to win. This reads every
submitted source the judge stored and counts the constructs present, so "skills raised Fortran
speed-up" can be checked against "the skilled Fortran arm used `do concurrent` and the other did
not", rather than assumed.

Detection is deliberately SYNTACTIC and conservative. A regex cannot know that a `collapse(2)` was
the thing that mattered, so nothing here claims attribution -- these are adoption rates, and the
speed-up tables stay the place where effect is measured. Constructs are matched on the token a
compiler would act on (a pragma, an intrinsic, a directive clause), never on a comment mentioning
one, which is why every pattern anchors on the directive form.

Usage:  python3 analysis/optimizations_llr8w2.py --run-root <runs> --out data-llr8w2
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import pathlib
import re
import sqlite3
import sys

#: job id -> arm, for the wave-2 arms. Same table as collect.py's ARMS; kept here as a mapping
#: because this script reads the blob store, which collect.py has no reason to touch.
ARMS = {
    610669: ("qwen38", "c", False),
    610671: ("qwen38", "c", True),
    610670: ("qwen38", "fortran", False),
    610672: ("qwen38", "fortran", True),
    610668: ("oss120b", "fortran", False),
    610653: ("oss120b", "fortran", True),
    610662: ("kimi27sglang", "c", True),
}

#: construct -> pattern. Ordered most specific first so a `simd` inside a `parallel for simd` is
#: counted as both, which is what a reader wants: the two say different things about the answer.
CONSTRUCTS: dict[str, re.Pattern] = {
    "omp_parallel": re.compile(r"^\s*[!#]\$?\s*(pragma\s+)?omp\s+(parallel|target)", re.M | re.I),
    "omp_simd": re.compile(r"^\s*[!#]\$?\s*(pragma\s+)?omp\s+[^\n]*\bsimd\b", re.M | re.I),
    "omp_collapse": re.compile(r"\bcollapse\s*\(", re.I),
    "omp_reduction": re.compile(r"\breduction\s*\(", re.I),
    "omp_schedule": re.compile(r"\bschedule\s*\(", re.I),
    "do_concurrent": re.compile(r"^\s*do\s+concurrent\b", re.M | re.I),
    "restrict": re.compile(r"\b(__restrict__|__restrict|\brestrict\b)"),
    "aligned": re.compile(r"\b(aligned\s*\(|__builtin_assume_aligned|assume_aligned|!dir\$\s+assume_aligned)", re.I),
    "unroll": re.compile(r"(#\s*pragma\s+(GCC\s+)?unroll|!dir\$\s+unroll|\bunroll\s*\()", re.I),
    "ivdep": re.compile(r"(ivdep|GCC\s+ivdep)", re.I),
    "blocking": re.compile(r"\b(tile|block|blk)[_a-z]*\s*=\s*\d+", re.I),
    "intrinsics": re.compile(r"\b(_mm\d*_|__m256|__m512|vec_)"),
    "math_reassoc": re.compile(r"(fast_math|ffast-math|reassoc|fma\s*\()", re.I),
    "array_syntax": re.compile(r"^\s*\w+\s*\(\s*:\s*[,)]", re.M),
    "matmul_intrinsic": re.compile(r"\b(matmul|dot_product)\s*\(", re.I),
}


def blob_root(db: str) -> pathlib.Path:
    """Where a shard's stored sources live: ``<stem>_prompts`` beside the database file."""
    path = pathlib.Path(db)
    return path.parent / f"{path.stem}_prompts"


def sources_of(db: str) -> list[tuple[str, str, str]]:
    """``(benchmark, language, text)`` for every source this shard stored and still has on disk."""
    out: list[tuple[str, str, str]] = []
    root = blob_root(db)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute("select benchmark, language, path from sources").fetchall()
    except sqlite3.DatabaseError:
        return out
    finally:
        con.close()
    for benchmark, language, rel in rows:
        blob = root / rel
        if blob.is_file():
            out.append((benchmark, language, blob.read_text(errors="replace")))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    per_arm: dict[str, dict[str, object]] = collections.OrderedDict()
    rows: list[dict[str, object]] = []
    for job, (model, language, skills) in ARMS.items():
        arm = f"llr8w2-{model}-{language}{'-skills' if skills else ''}"
        shards = sorted(glob.glob(str(args.run_root / str(job) / "judge" / "rank-*" / "hpcagent_bench*.db")))
        if not shards:
            print(f"{arm}: no shards under job {job}", file=sys.stderr)
            continue
        # One source per (benchmark, text): a kernel re-submitted unchanged is one answer, not two,
        # and counting it twice would inflate whichever construct that kernel happens to use.
        seen: set[tuple[str, str]] = set()
        counts: collections.Counter = collections.Counter()
        for shard in shards:
            for benchmark, _, text in sources_of(shard):
                key = (benchmark, text)
                if key in seen:
                    continue
                seen.add(key)
                for name, pattern in CONSTRUCTS.items():
                    if pattern.search(text):
                        counts[name] += 1
        answers = len(seen)
        per_arm[arm] = {"answers": answers, **{k: counts.get(k, 0) for k in CONSTRUCTS}}
        row = {"arm": arm, "model": model, "language": language, "skills": int(skills), "answers": answers}
        row.update({k: round(counts.get(k, 0) / answers, 4) if answers else 0.0 for k in CONSTRUCTS})
        rows.append(row)
        print(f"{arm}: {answers} distinct answers")

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "constructs.csv"
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["arm"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {target}  rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
