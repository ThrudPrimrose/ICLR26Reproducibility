"""compact_threshold_pack -- stream compaction, python (numba) implementation.

Packs src[i]*weight[i] for every src[i] > 0, in source order, into packed[:n],
and publishes n in out_count[0].

Strategy: two parallel passes over statically-mapped, contiguous per-thread
chunks (classic stream compaction):
  1. per-thread survivor counts (prange),
  2. short sequential exclusive prefix,
  3. per-thread parallel pack using the thread's base offset (prange).
Numba functions are warmed at import time so no compile cost lands in the
timed region; thread count is pinned to the process affinity.
"""

import os

import numba as nb
import numpy as np

# Honour the container affinity before anything parallel happens.
try:
    _AFF = len(os.sched_getaffinity(0))
except Exception:  # pragma: no cover
    _AFF = os.cpu_count() or 1
os.environ["NUMBA_NUM_THREADS"] = str(_AFF)
try:
    nb.set_num_threads(_AFF)
except Exception:
    pass


@nb.njit(parallel=True, cache=True, fastmath=True)
def _counts(src, counts, n):
    nt = nb.get_num_threads()
    for t in nb.prange(nt):
        lo = (n * t) // nt
        hi = (n * (t + 1)) // nt
        c = 0
        for i in range(lo, hi):
            c += (src[i] > 0.0)
        counts[t] = c


@nb.njit(cache=True, fastmath=True)
def _prefix_exclusive(counts):
    s = 0
    for i in range(len(counts)):
        c = counts[i]
        counts[i] = s
        s += c
    return s


@nb.njit(parallel=True, cache=True, fastmath=True)
def _pack(src, weight, packed, offs, n):
    nt = nb.get_num_threads()
    for t in nb.prange(nt):
        lo = (n * t) // nt
        hi = (n * (t + 1)) // nt
        base = offs[t]
        for i in range(lo, hi):
            if src[i] > 0.0:
                packed[base] = src[i] * weight[i]
                base += 1


def compact_threshold_pack(src, weight, packed, out_count, LEN_1D):
    n = int(LEN_1D)
    if n == 0:
        out_count[0] = 0
        return None
    nt = nb.get_num_threads()
    counts = np.empty(nt, dtype=np.int64)
    _counts(src, counts, n)
    total = _prefix_exclusive(counts)
    out_count[0] = total
    _pack(src, weight, packed, counts, n)
    return None


def _warm():
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 1000, 5000):
        src = rng.uniform(-1.0, 1.0, n)
        weight = rng.uniform(-1.0, 1.0, n)
        packed = np.zeros(n)
        out_count = np.zeros(1, dtype=np.int64)
        compact_threshold_pack(src, weight, packed, out_count, n)
        compact_threshold_pack(src, weight, packed, out_count, np.int64(n))


_warm()
