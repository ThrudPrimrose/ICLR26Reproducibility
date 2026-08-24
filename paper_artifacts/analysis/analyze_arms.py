"""Paired skills A/B per model x language, plus the coverage/quality figures.

Paired, not aggregate: the arms cover different kernel subsets, so an unpaired median
compares different populations and reads as an effect where there is none.
"""
import json
import math
import pathlib
import statistics

DATA = json.loads((pathlib.Path(__file__).with_name("arms.json")).read_text())

#: (label, no-skills job, skills job)
PAIRS = [
    ("qwen30b / C", "605458", "605459"),
    ("oss120b / C", "605460", "605461"),
    ("qwen30b / Fortran", "605696", "605697"),
]


def sign_test(win: int, lose: int) -> float:
    """Two-sided exact sign test. Ties are excluded, which is the standard treatment."""
    n = win + lose
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(win, lose) + 1))
    return min(1.0, 2 * tail / 2**n)


print(f"{'arm':22s} {'paired':>7s} {'skills+':>8s} {'skills-':>8s} {'tie':>5s} "
      f"{'p':>7s} {'med no':>7s} {'med sk':>7s}")
for label, a, b in PAIRS:
    na, sk = DATA[a]["best"], DATA[b]["best"]
    both = sorted(set(na) & set(sk))
    win = sum(1 for k in both if sk[k] > na[k] + 1e-9)
    lose = sum(1 for k in both if sk[k] < na[k] - 1e-9)
    tie = len(both) - win - lose
    print(f"{label:22s} {len(both):7d} {win:8d} {lose:8d} {tie:5d} "
          f"{sign_test(win, lose):7.3f} "
          f"{statistics.median(na[k] for k in both):7.3f} "
          f"{statistics.median(sk[k] for k in both):7.3f}")

print()
print(f"{'arm':30s} {'kernels':>8s} {'subs':>6s} {'ok%':>6s} {'incorrect%':>11s} "
      f"{'>1.0x':>7s} {'med':>7s}")
for job in sorted(DATA, key=lambda j: (DATA[j]["model"], DATA[j]["language"], DATA[j]["skills"])):
    d = DATA[job]
    st = d["status"]
    tot = sum(st.values()) or 1
    best = d["best"]
    gain = sum(1 for v in best.values() if v > 1.0)
    name = f"{d['model']}/{d['language']}{'+skills' if d['skills'] else ''}"
    med = statistics.median(best.values()) if best else float("nan")
    print(f"{name:30s} {len(best):8d} {d['submissions']:6d} "
          f"{100*st.get('ok',0)/tot:6.1f} {100*st.get('incorrect',0)/tot:11.1f} "
          f"{gain:3d}/{len(best):<3d} {med:7.3f}")
