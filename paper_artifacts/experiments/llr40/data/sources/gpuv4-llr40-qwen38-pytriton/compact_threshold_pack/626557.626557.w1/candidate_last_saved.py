"""Optimized stream compaction (compact_threshold_pack).

Reference: for every i with src[i] > 0, packed[n] = src[i]*weight[i], n+=1;
publish out_count[0] = n.  The write cursor is a loop-carried value, so the
baseline is a serial dependent loop.

Strategy:
  * large inputs: 3-pass OpenMP (numba prange) parallel compaction:
      pass 1: per-chunk survivor counts
      pass 2: serial exclusive scan over the small chunk counts
      pass 3: per-chunk packing at the scanned offsets
  * small inputs: single-thread numba kernel (call overhead << numpy ops)
All JIT compilation happens at import time, outside the timed section.
"""

import os

import numpy as np

try:
    import numba
    from numba import njit, prange

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False


if _HAVE_NUMBA:

    @njit(fastmath=True)
    def _compact_serial(src, weight, packed, out_count, n):
        c = 0
        for i in range(n):
            s = src[i]
            if s > 0.0:
                packed[c] = s * weight[i]
                c += 1
        out_count[0] = c

    @njit(parallel=True, fastmath=True)
    def _compact_par(src, weight, packed, out_count, n, nt):
        counts = np.empty(nt, dtype=np.int64)
        for i in prange(nt):
            lo = i * n // nt
            hi = (i + 1) * n // nt
            c = 0
            for j in range(lo, hi):
                if src[j] > 0.0:
                    c += 1
            counts[i] = c
        off = np.empty(nt + 1, dtype=np.int64)
        off[0] = 0
        tot = 0
        for i in range(nt):
            tot += counts[i]
            off[i + 1] = tot
        out_count[0] = tot
        for i in prange(nt):
            lo = i * n // nt
            hi = (i + 1) * n // nt
            c = 0
            base = off[i]
            for j in range(lo, hi):
                s = src[j]
                if s > 0.0:
                    packed[base + c] = s * weight[j]
                    c += 1


    def _cgroup_threads():
        # cgroup v2
        try:
            with open("/sys/fs/cgroup/cpu.max") as f:
                p = f.read().split()
            if p and p[0] != "max":
                return int(p[0]) // max(1, int(p[1]))
        except Exception:
            pass
        # cgroup v1
        try:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
                q = int(f.read())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                per = int(f.read())
            if q > 0:
                return q // per
        except Exception:
            pass
        return os.cpu_count() or 1

    def _affinity():
        try:
            return len(os.sched_getaffinity(0))
        except Exception:
            return os.cpu_count() or 1

    _CPU = min(_affinity(), os.cpu_count() or 1)
    _CGROUP = _cgroup_threads()
    _want = min(_CPU, _CGROUP) if _CGROUP > 0 else _CPU
    _NT = _want
    for _t in (_want, _want // 2, _want // 4, 1):
        if _t < 1:
            _t = 1
        try:
            numba.set_num_threads(_t)
            _NT = _t
            break
        except Exception:
            continue
    # chunks per invocation (allow load balancing beyond core count)
    _CHUNKS = 8 * _NT
    # warm the JIT now, at import time
    _d = np.ones(8192, dtype=np.float64)
    _o = np.zeros(8192, dtype=np.float64)
    _c = np.zeros(1, dtype=np.int64)
    _compact_serial(_d, _d, _o, _c, 8192)
    _compact_par(_d, _d, _o, _c, 8192, _CHUNKS)
    del _d, _o, _c


def _fallback(src, weight, packed, out_count, LEN_1D):
    n = int(LEN_1D)
    mask = src[:n] > 0.0
    k = int(mask.sum())
    if k:
        packed[:k] = src[:n][mask] * weight[:n][mask]
    out_count[0] = k


def compact_threshold_pack(src, weight, packed, out_count, LEN_1D):
    n = int(LEN_1D)
    if _HAVE_NUMBA and n >= (1 << 14) and \
            src.dtype == np.float64 and weight.dtype == np.float64 and \
            packed.dtype == np.float64 and \
            src.flags["C_CONTIGUOUS"] and weight.flags["C_CONTIGUOUS"] and \
            packed.flags["C_CONTIGUOUS"]:
        _compact_par(src, weight, packed, out_count, n, _CHUNKS)
    elif _HAVE_NUMBA and n < (1 << 14) and \
            src.dtype == np.float64 and weight.dtype == np.float64 and \
            packed.dtype == np.float64 and \
            src.flags["C_CONTIGUOUS"] and weight.flags["C_CONTIGUOUS"] and \
            packed.flags["C_CONTIGUOUS"]:
        _compact_serial(src, weight, packed, out_count, n)
    else:
        _fallback(src, weight, packed, out_count, n)
