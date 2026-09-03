"""Intervention efficacy: how much a skills packet (or any paired treatment) moves an arm.

Three axes, each a ratio of after to before, each 1.0 at no effect:

    rho_c  completion   Jeffreys-smoothed solve-count ratio      (extensive margin)
    rho_S  score        geomean of S_i / C_i over SOLVED kernels (intensive margin)
    rho_T  cost         tokens BEFORE / tokens AFTER             (inverted: >1 is better)

C_i is the per-kernel ceiling, the best verified score known at this dataset revision.
It cancels exactly when both arms solved the same kernels; where the solve sets differ it
removes the difficulty confound, since the arm that solved more is otherwise credited for
whichever kernels happened to be easy.

Two scales are reported for the composite, because the choice was live during drafting:

    LN  sum of w * ln(rho)      -- additive under composition, log-scaled tail
    G   sum of w * g(rho)       -- g(2)=+1, g(0.5)=-1, linear tail

Both are antisymmetric: naming either arm the baseline flips the sign and nothing else.
LN is the one to publish -- it composes (two successive 2x gains sum to ln 4 exactly, so a
suite total exponentiates back into an "overall Nx" statement) and its tail is logarithmic,
which matters because a single 104x cell has already distorted one arm of the git experiment.
G is kept because it reads more naturally, and on this data the two agree to within 0.04.

NEITHER is ever applied to a raw per-cell speed-up, only to the three leg-level ratios: g is
linear in the ratio, so one 100x cell would otherwise carry an entire sum.
"""

import csv
import math
import os
import random
from collections import defaultdict

CSV = os.environ.get("EFFICACY_CSV", "experiments/llr9/data/kernels.csv")
W = (1 / 3, 1 / 3, 1 / 3)  # w_c, w_S, w_T


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 1.0


def g(rho):
    """Symmetric relative change: g(2) = +1, g(0.5) = -1, g(1) = 0, g(1/r) = -g(r).
    Continuous and smooth at 1, unlike the bare signed fold change, which leaps +1 to -1."""
    return rho - 1.0 if rho >= 1.0 else 1.0 - 1.0 / rho


def gains(cells, ceil=None):
    """{kernel: {'0': (solved, S, tokens), '1': ...}} -> (rho_c, rho_S, rho_T)."""
    ceil = ceil or {}
    solves = {a: sum(v[a][0] for v in cells.values() if a in v) for a in "01"}
    tok = {a: sum(v[a][2] for v in cells.values() if a in v) for a in "01"}
    rho_c = (solves["1"] + 0.5) / (solves["0"] + 0.5)  # Jeffreys: finite at 0 and at a clean sweep
    # Score CONDITIONAL on solving, so completion is not counted twice: an unsolved task
    # already scores 1.0, and a roster-wide geomean would therefore re-encode the solve count.
    s = {
        a: geomean([v[a][1] / ceil.get(k, 1.0) for k, v in cells.items() if a in v and v[a][0]])
        for a in "01"
    }
    rho_S = s["1"] / s["0"] if s["0"] > 0 else 1.0
    rho_T = tok["0"] / tok["1"] if tok["1"] > 0 else 1.0
    return rho_c, rho_S, rho_T


def efficacy(cells, w=W, ceil=None, scale="ln"):
    """(Q, (rho_c, rho_S, rho_T)). Q = 0 at no effect, positive when the treatment helped."""
    r = gains(cells, ceil)
    f = math.log if scale == "ln" else g
    return sum(wi * f(ri) for wi, ri in zip(w, r)), r


def bootstrap(cells, n=4000, seed=0, ceil=None, scale="ln"):
    """Paired percentile bootstrap over the kernel roster. A CI covering 0 means no effect."""
    rng = random.Random(seed)
    keys = list(cells)
    out = []
    for _ in range(n):
        draw = [rng.choice(keys) for _ in keys]
        out.append(efficacy({f"{i}:{k}": cells[k] for i, k in enumerate(draw)}, ceil=ceil, scale=scale)[0])
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n)]


def ceilings(legs):
    """Best score per kernel over every arm. In the harness this must be the max over
    submissions that passed independent_verify AND were not flagged suspect, over BOTH agent
    and non-AI optimizers, and FROZEN with the dataset revision: an unpinned ceiling silently
    re-scores published results."""
    c = defaultdict(lambda: 1.0)
    for cells in legs.values():
        for k, v in cells.items():
            for arm in v.values():
                if arm[0]:
                    c[k] = max(c[k], arm[1])
    return dict(c)


def load(path=CSV):
    legs = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        legs[(r["model"], r["language"])].setdefault(r["benchmark"], {})[r["skills"]] = (
            int(r["solved"]),
            max(float(r["last_speedup"] or 0), 1.0),  # the S_i convention: unsolved scores 1.0
            int(float(r["tokens"] or 0)),
        )
    return legs


def paired(legs, key):
    return {k: v for k, v in legs[key].items() if "0" in v and "1" in v}


if __name__ == "__main__":
    legs = load()
    ceil = ceilings(legs)
    print(
        f"ceiling coverage: {sum(1 for v in ceil.values() if v > 1.0)}/{len(ceil)} kernels above 1.0x, "
        f"geomean {geomean(list(ceil.values())):.1f}x\n"
    )
    hdr = f"{'leg':24s} {'rho_c':>7s} {'rho_S':>7s} {'rho_T':>7s} | {'Q(ln)':>7s} {'95% CI':>18s} | {'Q(g)':>7s} {'gap':>6s}"
    print(hdr)
    qs_ln, qs_g = [], []
    for key in sorted(legs):
        cells = paired(legs, key)
        if not cells:
            continue
        q_ln, (rc, rs, rt) = efficacy(cells, ceil=ceil, scale="ln")
        q_g, _ = efficacy(cells, ceil=ceil, scale="g")
        lo, hi = bootstrap(cells, ceil=ceil, scale="ln")
        qs_ln.append(q_ln)
        qs_g.append(q_g)
        flag = "" if lo <= 0.0 <= hi else " *"
        print(
            f"{key[0]+'/'+key[1]:24s} {rc:7.3f} {rs:7.3f} {rt:7.3f} | {q_ln:+7.4f} "
            f"[{lo:+.3f},{hi:+.3f}]{flag:2s} | {q_g:+7.4f} {abs(q_ln-q_g):6.4f}"
        )
    n = len(qs_ln)
    print(f"\nsuite Q(ln) = {sum(qs_ln)/n:+.4f}  -> overall {math.exp(sum(qs_ln)/n):.3f}x effect")
    print(f"suite Q(g)  = {sum(qs_g)/n:+.4f}")
    same_sign = sum((a > 0) == (b > 0) for a, b in zip(qs_ln, qs_g))
    order_ln = [i for i, _ in sorted(enumerate(qs_ln), key=lambda t: -t[1])]
    order_g = [i for i, _ in sorted(enumerate(qs_g), key=lambda t: -t[1])]
    print(f"\nln vs g: {same_sign}/{n} legs agree on sign; ordering {'IDENTICAL' if order_ln == order_g else 'DIFFERS'}")
    print("* = 95% CI excludes 0 (a real effect); every other leg is indistinguishable from no effect")
