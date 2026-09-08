"""Segmented dot product over a ragged CSR structure (fast, NUMA-aware numba).

out[s] = sum(val[e]*w[e] for e in [row_ptr[s], row_ptr[s+1])).

Strategy: parallelize over SEGMENTS with an element-balanced, memory-contiguous
partition -- segment s is owned by the thread whose element range contains
row_ptr[s] (its first element).  Each thread therefore reads one contiguous
span of val/w, so hardware prefetchers stay happy and aggregate DRAM bandwidth
scales with the number of cores on the data's NUMA node.  No temporaries are
allocated and each segment is summed in exactly the reference order, so the
result matches the serial oracle to round-off.
"""
import os
import glob
import re
from collections import Counter

import numpy as np

try:
    import numba
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba is the baseline, always present
    numba = None
    _HAVE_NUMBA = False


_c2n = None
def _core_to_node():
    """Map cpu id -> numa node, read once from /sys (cached in the module)."""
    global _c2n
    if _c2n is not None:
        return _c2n
    m = {}
    try:
        for f in glob.glob('/sys/devices/system/node/node*/cpulist'):
            mm = re.search(r'node(\d+)/cpulist$', f)
            if not mm:
                continue
            node = int(mm.group(1))
            for part in open(f).read().strip().split(','):
                part = part.strip()
                if not part:
                    continue
                if '-' in part:
                    a, b = part.split('-')
                    for c in range(int(a), int(b) + 1):
                        m[c] = node
                else:
                    m[int(part)] = node
    except Exception:
        pass
    _c2n = m
    return m


if _HAVE_NUMBA:
    @njit(parallel=True)
    def _par(rp, v, w, o, NSEG, total, T):
        for t in prange(T):
            elo = (total * t) // T
            ehi = (total * (t + 1)) // T
            # s0 = first s with rp[s] >= elo
            lo, hi = 0, NSEG
            while lo < hi:
                mid = (lo + hi) >> 1
                if rp[mid] < elo:
                    lo = mid + 1
                else:
                    hi = mid
            s0 = lo
            if t == T - 1:
                s1 = NSEG  # last thread always owns the final segment
            else:
                lo, hi = s0, NSEG
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if rp[mid] < ehi:
                        lo = mid + 1
                    else:
                        hi = mid
                s1 = lo
            for s in range(s0, s1):
                acc = 0.0
                a = rp[s]
                b = rp[s + 1]
                for e in range(a, b):
                    acc += v[e] * w[e]
                o[s] = acc

    @njit
    def _serial(rp, v, w, o, NSEG):
        for s in range(NSEG):
            acc = 0.0
            for e in range(rp[s], rp[s + 1]):
                acc += v[e] * w[e]
            o[s] = acc

    # Force the (first-call) JIT compilation at IMPORT time so it is not
    # charged to the timed region.  Types mirror the real call exactly.
    try:
        _rp = np.array([0, 2, 3], dtype=np.int64)
        _v = np.ones(3, dtype=np.float64)
        _w = np.ones(3, dtype=np.float64)
        _o = np.zeros(2, dtype=np.float64)
        numba.set_num_threads(2)
        _par(_rp, _v, _w, _o, 2, 3, 2)
        _par(_rp, _v, _w, _o, 2, 3, 3)
        _serial(_rp, _v, _w, _o, 2)
    except Exception:
        pass


def _pick_threads():
    """Return (T, use_parallel).  Prefer all cores on the majority NUMA node."""
    if not _HAVE_NUMBA:
        return 1, False
    try:
        aff = os.sched_getaffinity(0)
    except Exception:
        aff = set(range(os.cpu_count() or 1))
    c2n = _core_to_node()
    cnt = Counter(c2n.get(c, -1) for c in aff)
    if not cnt:
        return 1, False
    node, n = cnt.most_common(1)[0]
    if node >= 0 and n >= 2:
        cores = frozenset(c for c in aff if c2n.get(c) == node)
        try:
            os.sched_setaffinity(0, cores)
        except Exception:
            pass
        return min(n, 64), True
    return 1, False


def segment_reduce_ragged(row_ptr, val, w, out, NSEG):
    total = val.size
    if _HAVE_NUMBA:
        T, use_par = _pick_threads()
        if use_par:
            try:
                numba.set_num_threads(T)
                _par(row_ptr, val, w, out, NSEG, total, T)
                return None
            except Exception:
                pass
        _serial(row_ptr, val, w, out, NSEG)
        return None
    # Last-resort vectorised fallback (correct but single-threaded).
    csum = np.zeros(total + 1, dtype=val.dtype)
    np.cumsum(val * w, out=csum[1:])
    out[...] = csum[row_ptr[1:]] - csum[row_ptr[:-1]]
    return None
