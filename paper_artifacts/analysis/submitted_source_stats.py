"""Tally what agents actually WROTE, per arm, from the agent transcripts.

The judge DBs record the verdict but never the source, and score/submit take a PATH rather
than the text, so construct adoption has to come from the Write/Edit tool inputs. Counted
once per worker: did this agent's code ever contain the construct.
"""
import collections
import json
import pathlib
import re

ARMS = {"qwen30b/C no-skills": 605458, "qwen30b/C skills": 605459,
        "oss120b/C no-skills": 605460, "oss120b/C skills": 605461}

PATTERNS = {
    "omp parallel": re.compile(r"#pragma\s+omp\s+parallel"),
    "omp simd": re.compile(r"omp\s+(parallel\s+for\s+)?simd"),
    "reduction": re.compile(r"reduction\s*\("),
    "collapse": re.compile(r"collapse\s*\("),
    "restrict": re.compile(r"\brestrict\b"),
    "ivdep/assume": re.compile(r"ivdep|__builtin_assume"),
    "unroll": re.compile(r"#pragma\s+(GCC\s+)?unroll"),
    "tiling": re.compile(r"\b(ii|jj|kk)\b"),
}


def written_text(log: pathlib.Path) -> str:
    """Every blob this agent wrote into a file, concatenated."""
    out = []
    for line in log.read_text(errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        for block in ((rec.get("message") or {}).get("content") or []):
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            if (block.get("name") or "") not in ("Write", "Edit", "MultiEdit"):
                continue
            inp = block.get("input") or {}
            for key in ("content", "new_string"):
                val = inp.get(key)
                if isinstance(val, str):
                    out.append(val)
            for ed in (inp.get("edits") or []):
                if isinstance(ed, dict) and isinstance(ed.get("new_string"), str):
                    out.append(ed["new_string"])
    return "\n".join(out)


def main() -> None:
    header = " ".join(f"{k[:11]:>12s}" for k in PATTERNS)
    print(f"{'arm':24s} {'agents':>6s} {header}")
    for label, job in ARMS.items():
        root = pathlib.Path(f"/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs/{job}/agents")
        hits: collections.Counter = collections.Counter()
        n = 0
        for log in root.glob("node-*/problem-*-worker-*/claude.log"):
            text = written_text(log)
            if len(text) < 40:
                continue
            n += 1
            for key, rx in PATTERNS.items():
                if rx.search(text):
                    hits[key] += 1
        if not n:
            print(f"{label:24s} {0:6d}  (nothing recovered)")
            continue
        cells = " ".join(f"{100 * hits[k] / n:11.0f}%" for k in PATTERNS)
        print(f"{label:24s} {n:6d} {cells}")


if __name__ == "__main__":
    main()
