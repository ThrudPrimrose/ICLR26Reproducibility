import os
os.environ.setdefault("NUMBA_NUM_THREADS", "48")
import time, sys
import numpy as np
import numba
from numba import njit, prange, set_num_threads

def topo():
    try:
        sib = open("/sys/devices/system/cpu/cpu24/topology/thread_siblings_list").read().strip()
        pkg = open("/sys/devices/system/cpu/cpu24/topology/physical_package_id").read().strip()
        l3 = open("/sys/devices/system/cpu/cpu24/cache/index3/size").read().strip()
        return f"sib24={sib} pkg={pkg} L3={l3}"
    except Exception as e:
        return f"topo fail {e!r}"

@njit
def ser(a, b, c, n):
    t = 0.0
    for i in range(n):
        s = b[i] * c[i]
        a[i] = s + t
        t = s

@njit(parallel=True)
def par(a, b, c, n):
    for i in prange(n):
        s = b[i] * c[i]
        if i > 0:
            s += b[i - 1] * c[i - 1]
        a[i] = s

ser(np.zeros(0), np.zeros(0), np.zeros(0), 0)
par(np.zeros(0), np.zeros(0), np.zeros(0), 0)

def s252(a, b, c, LEN_1D):
    n = int(LEN_1D)
    out = [f"aff={len(os.sched_getaffinity(0))}", topo()]
    try:
        t0 = time.perf_counter(); ser(a, b, c, n); out.append(f"ser={((time.perf_counter()-t0)*1e3):.1f}ms")
        for k in (24, 32, 48, 24):
            set_num_threads(k)
            par(a, b, c, n)
            ts = []
            for _ in range(3):
                t0 = time.perf_counter(); par(a, b, c, n); ts.append((time.perf_counter() - t0) * 1e3)
            out.append(f"par{k}=" + "/".join(f"{t:.1f}" for t in ts))
        set_num_threads(24)
    finally:
        print(" | ".join(out)); sys.stdout.flush()
        par(a, b, c, n)
    return None
