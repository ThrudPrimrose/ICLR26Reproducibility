"""TSVC tsvc_2 ``s311`` -- sum a[0:LEN_1D] into sum_out[0] (in-place, returns None).

v3: multi-threaded chunked NumPy pairwise sums. A persistent thread pool is
created at import time (untimed); each worker sums a contiguous chunk with
np.sum (releases the GIL). Thread count adapts to affinity and cgroup quota.
"""
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def _cpu_budget():
    try:
        n = len(os.sched_getaffinity(0))
    except Exception:
        n = os.cpu_count() or 1
    for path in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        try:
            with open(path) as fh:
                tok = fh.read().split()
            if path.endswith("cpu.max"):
                if tok and tok[0] != "max":
                    n = min(n, max(1, int(float(tok[0]) / float(tok[1]))))
            else:
                if tok and tok[0] != "-1":
                    n = min(n, max(1, int(int(tok[0]) / int(tok[1]))))
            break
        except Exception:
            continue
    return max(1, n)


_T = _cpu_budget()
_POOL = ThreadPoolExecutor(max_workers=_T)


def _chunk_sum(a, i0, i1):
    return float(np.sum(a[i0:i1]))


def s311(a, sum_out, LEN_1D):
    n = int(LEN_1D)
    if n <= 0:
        sum_out[0] = 0.0
        return None
    T = _T if _T < n else n
    if T == 1:
        sum_out[0] = np.sum(a[:n])
        return None
    base, rem = divmod(n, T)
    # boundaries for T equal chunks (first `rem` chunks get one extra)
    starts = [0] * (T + 1)
    s = 0
    for t in range(T):
        s += base + (1 if t < rem else 0)
        starts[t + 1] = s
    total = 0.0
    futs = [_POOL.submit(_chunk_sum, a, starts[t], starts[t + 1]) for t in range(T)]
    for f in futs:
        total += f.result()
    sum_out[0] = total
    return None


tsvc_2_s311_fp64 = s311

# Warm the pool (untimed import-time work).
_w = np.random.default_rng(0).standard_normal(1 << 20)
_ws = np.zeros(1)
s311(_w, _ws, _w.shape[0])
s311(_w[:17], _ws, 17)
del _w, _ws
