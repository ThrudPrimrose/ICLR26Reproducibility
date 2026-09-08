"""Optimized versioned_distance_update.

a[i] = 0.75 * a[i-K] + b[i] * c[i], in-place on a.

The dependence distance K is a runtime symbol (domain {1, 5, 64, 251}); there are K
independent chains (chain r = indices r, r+K, r+2K, ...).  The carry decays by 0.75
per step, so after S = 256 steps the carry is 0.75**256 ~ 1e-32: a chunk of S chain
steps only needs the previous chunk's carry, which equals that chunk's zero-init
end value to ~1e-32 absolute.  Hence a two-phase blocked scan:

  phase A (parallel over all (chunk, chain) blocks): zero-init scan of each chunk.
  phase C (parallel over the same blocks): real scan, chunk 0 seeded from a[0:K],
          chunk k >= 1 initialized with the previous chunk's phase-A value.

Both phases touch every element once; no per-thread scratch; b, c are read inline
(no temp product array), a is written in place (only the K seeds need saving).
"""
import numpy as np
import numba as nb
from numba import njit, prange

_S = 256
_SMALL_N = 1 << 17


@njit(cache=True)
def _serial(a, b, c, n, K):
    for i in range(K, n):
        a[i] = 0.75 * a[i - K] + b[i] * c[i]


@njit(parallel=True, fastmath=True, cache=True)
def _phase_a(b, c, P, n, K, S, C):
    for blk in prange(C * K):
        r = blk % K
        k = blk // K
        p = 0.0
        base = k * S * K + r
        for s in range(S):
            idx = base + s * K
            if idx < n:
                p = 0.75 * p + b[idx] * c[idx]
        P[blk] = p


@njit(parallel=True, fastmath=True, cache=True)
def _phase_c(a, b, c, seeds, P, n, K, S, C):
    for blk in prange(C * K):
        r = blk % K
        k = blk // K
        if k == 0:
            prev = seeds[r]
            s0 = 1  # position r (m=0) is the seed; untouched
        else:
            prev = P[blk - K]
            s0 = 0
        base = k * S * K + r
        for s in range(s0, S):
            idx = base + s * K
            if idx < n:
                v = 0.75 * prev + b[idx] * c[idx]
                a[idx] = v
                prev = v


def versioned_distance_update(a, b, c, LEN_1D, K):
    n = int(LEN_1D)
    K = int(K)
    if K >= n:
        return None
    if n < _SMALL_N:
        _serial(a, b, c, n, K)
        return None
    seeds = a[:K].copy()
    Lmax = (n + K - 1) // K
    C = (Lmax + _S - 1) // _S
    P = np.empty(C * K, dtype=np.float64)
    _phase_a(b, c, P, n, K, _S, C)
    _phase_c(a, b, c, seeds, P, n, K, _S, C)
    return None


def _warm():
    n = 3 * _S + 7
    a = np.random.rand(n)
    b = np.random.rand(n)
    c = np.random.rand(n)
    for K in (1, 5, 64, 251):
        if K < n:
            _serial(a.copy(), b, c, n, K)
            seeds = a[:K].copy()
            Lmax = (n + K - 1) // K
            C = (Lmax + _S - 1) // _S
            P = np.empty(C * K)
            _phase_a(b, c, P, n, K, _S, C)
            _phase_c(a, b, c, seeds, P, n, K, _S, C)


_warm()
