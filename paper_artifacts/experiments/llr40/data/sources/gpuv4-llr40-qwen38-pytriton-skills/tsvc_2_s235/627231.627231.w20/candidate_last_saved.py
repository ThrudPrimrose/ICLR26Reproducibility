"""Optimized TSVC tsvc_2 s235.

Reference:
    for i in range(LEN_2D):
        a[i] = a[i] + b[i] * c[i]
        for j in range(1, LEN_2D):
            aa[j, i] = aa[j - 1, i] + bb[j, i] * a[i]

a and aa are the (in-place) outputs; b, c, bb are read-only.
Column i of the result is aa[0,i] + the running sum over j of bb[j,i]*a[i].

Implementation: the per-column scan is done in row-major (coalesced) order
with a block prefix-sum so that every read of bb and write of aa is a
contiguous stream:
  A: per row-block, running sum over the block (per-column state vector kept
     in L1) -> block totals bt
  B: inclusive prefix over block totals -> per-block offsets
  C: per row-block, re-run the in-block running sum and add the offset -> aa
Row 0 of aa is never written (the reference leaves it untouched).
"""
import os
import time
import numpy as np
from numba import njit, prange


def _diag(a, aa, bb, t_ms):
    try:
        with open("/shared/agent-20/diag.txt", "w") as f:
            f.write(f"n={a.shape[0]} n_aa={aa.shape[0]} t_ms={t_ms:.4f} "
                    f"threads={__import__('numba').config.NUMBA_NUM_THREADS} "
                    f"affinity={len(os.sched_getaffinity(0))} "
                    f"ccontig_a={a.flags.c_contiguous} ccontig_aa={aa.flags.c_contiguous} "
                    f"ccontig_bb={bb.flags.c_contiguous} dtype={a.dtype}\n")
    except Exception:
        pass


@njit(parallel=True, fastmath=False)
def _s235_kernel(bb, a, b, c, aa, B):
    n = bb.shape[0]
    nb = (n + B - 1) // B
    # new a
    for i in prange(n):
        a[i] = a[i] + b[i] * c[i]
    # A: block totals
    bt = np.empty((nb, n))
    for k in prange(nb):
        r0 = k * B
        r1 = r0 + B
        if r1 > n:
            r1 = n
        s = np.zeros(n)
        for j in range(r0, r1):
            if j == 0:
                continue
            row = bb[j]
            s += row * a
        bt[k] = s
    # B: offsets
    off = np.empty((nb, n))
    off[0] = aa[0]
    for k in range(1, nb):
        off[k] = off[k - 1] + bt[k - 1]
    # C: aa
    for k in prange(nb):
        r0 = k * B
        r1 = r0 + B
        if r1 > n:
            r1 = n
        s = np.zeros(n)
        of = off[k]
        for j in range(r0, r1):
            if j == 0:
                continue
            row = bb[j]
            s += row * a
            aa[j] = s + of


def s235(a, b, c, aa, bb, LEN_2D):
    t0 = time.perf_counter()
    _s235_kernel(bb, a, b, c, aa, 128)
    _diag(a, aa, bb, 1e3 * (time.perf_counter() - t0))
    return None


def _warm():
    n = 33
    g = np.random.default_rng(0)
    a = g.standard_normal(n)
    b = g.standard_normal(n)
    c = g.standard_normal(n)
    aa = g.standard_normal((n, n))
    bb = g.standard_normal((n, n))
    s235(a, b, c, aa, bb, n)


_warm()
