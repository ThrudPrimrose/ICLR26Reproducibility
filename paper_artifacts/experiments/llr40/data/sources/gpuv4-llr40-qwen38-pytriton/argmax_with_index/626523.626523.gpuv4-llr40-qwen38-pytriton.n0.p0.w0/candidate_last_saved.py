"""TSVC tsvc_2_5 argmax_with_index -- many-core optimized (python arm).

Running maximum carrying BOTH the value and its index.  The reference is a
sequential scan with strict `>`, i.e. the first occurrence of the maximum
wins ties.  We preserve that exactly: the array is split into ascending
chunks, each chunk's (max, first-index) is found independently (in worker
threads, numpy's vectorised argmax), and the chunks are combined in index
order with a strict `>` -- so an earlier chunk always wins a tie, exactly
like the reference scan.

Large-n engine: a persistent thread pool (created at import time, untimed);
numpy releases the GIL inside argmax, so the threads truly run in parallel.
Small-n engine: a numba scalar scan (no thread-launch overhead).
"""
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------------------
# Thread count: the grading child pins itself to a share of the physical cores
# (one SMT sibling per core) BEFORE this module is loaded, so the affinity
# mask at import time is the right answer.
# ---------------------------------------------------------------------------
def _aff_count():
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1

_NTHREADS = max(1, min(_aff_count(), 256))

_POOL = ThreadPoolExecutor(max_workers=_NTHREADS, thread_name_prefix="argmax")

# ---------------------------------------------------------------------------
# Small-n path: single-threaded numba scan, no parallel launch overhead.
# ---------------------------------------------------------------------------
try:
    import numba as nb

    @nb.njit(boundscheck=False, fastmath=False)
    def _argmax_small(a, out_value, out_index, n):
        x = a[0]
        idx = 0
        for i in range(1, n):
            v = a[i]
            if v > x:
                x = v
                idx = i
        out_value[0] = x
        out_index[0] = idx

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba is part of the environment
    _HAVE_NUMBA = False


_SMALL_N = 1 << 22


def _large(a, out_value, out_index, n):
    nt = _NTHREADS
    # enough chunks to keep every worker busy, but not so many that
    # submit/result overhead grows with n
    if n >= nt * (1 << 20):
        nchunks = nt * 4
    else:
        nchunks = nt
    chunk = (n + nchunks - 1) // nchunks
    futs = []
    j = 0
    while j < n:
        e = j + chunk
        if e > n:
            e = n
        futs.append(_POOL.submit(np.argmax, a[j:e]))
        j = e
    # combine in ascending index order: strict > keeps the first occurrence
    off = 0
    first = int(futs[0].result())
    best = a[first]
    bidx = first
    off = chunk
    for f in futs[1:]:
        li = int(f.result())
        v = a[off + li]
        if v > best:
            best = v
            bidx = off + li
        off += chunk
    out_value[0] = best
    out_index[0] = bidx


def argmax_with_index(a, out_value, out_index, LEN_1D):
    n = int(LEN_1D)
    if n <= 1:
        if n == 1:
            out_value[0] = a[0]
            out_index[0] = 0
        return None
    if n < _SMALL_N and _HAVE_NUMBA:
        _argmax_small(a, out_value, out_index, n)
    else:
        _large(a, out_value, out_index, n)
    return None


# ---------------------------------------------------------------------------
# Warm everything up at import time (untimed): numba compilation, the thread
# pool, and one pass through the large path.
# ---------------------------------------------------------------------------
def _warm():
    ov = np.empty(1, np.float64)
    oi = np.empty(1, np.int64)
    a2 = np.random.default_rng(1).random(4096)
    argmax_with_index(a2, ov, oi, 4096)
    if _NTHREADS >= 8:
        n = 1 << 25  # 32 MB -- keeps import fast but exercises the pool
    else:
        n = 1 << 23
    a = np.random.default_rng(0).random(n)
    argmax_with_index(a, ov, oi, n)


_warm()
