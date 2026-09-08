import numpy as np
from numba import njit, prange

_H = 256


@njit(parallel=True, fastmath=True)
def _scan(out, cc, N, H):
    # For every column c in [8, N): out[r, c] for r >= 8 becomes
    #   out[7, c] + sum_{r0=8..r} cc[r0, c]
    # Two-level scan along the row axis: chunk sums, prefix over chunks,
    # then in-chunk cumsum. prange over chunks (independent once prefix known).
    M = N - 8
    if M <= 0:
        return
    Nc = (M + H - 1) // H
    S = np.empty((Nc, M), dtype=out.dtype)
    # L1: per-chunk sums
    for ch in prange(Nc):
        r0 = 8 + ch * H
        r1 = r0 + H
        if r1 > N:
            r1 = N
        srow = S[ch]
        for k in range(M):
            srow[k] = 0.0
        for r in range(r0, r1):
            crow = cc[r]
            for k in range(M):
                srow[k] += crow[k + 8]
    # L2: prefix over chunk sums (short serial chain)
    for ch in range(1, Nc):
        sp = S[ch - 1]
        sc = S[ch]
        for k in range(M):
            sc[k] += sp[k]
    # L3: in-chunk cumsum + chunk prefix
    seed = out[7]
    for ch in prange(Nc):
        r0 = 8 + ch * H
        r1 = r0 + H
        if r1 > N:
            r1 = N
        t = np.empty(M, dtype=out.dtype)
        if ch == 0:
            for k in range(M):
                t[k] = seed[k + 8]
        else:
            sp = S[ch - 1]
            for k in range(M):
                t[k] = seed[k + 8] + sp[k]
        for r in range(r0, r1):
            crow = cc[r]
            orow = out[r]
            for k in range(M):
                t[k] += crow[k + 8]
                orow[k + 8] = t[k]


def s2233(aa, bb, cc, LEN_2D):
    import sys
    print("PROBE", aa.shape, aa.dtype, cc.dtype, aa.strides, aa.flags.c_contiguous, type(LEN_2D).__name__, int(LEN_2D), file=sys.stdout)
    sys.stdout.flush()
    N = int(LEN_2D)
    _scan(aa, cc, N, _H)
    _scan(bb, cc, N, _H)
    return None


# Warm up at import time (untimed): compile the variants we expect.
def _warm():
    for dt in (np.float64, np.float32):
        N = 64
        a = np.ones((N, N), dt)
        c = np.ones((N, N), dt)
        _scan(a, c, N, _H)


_warm()
