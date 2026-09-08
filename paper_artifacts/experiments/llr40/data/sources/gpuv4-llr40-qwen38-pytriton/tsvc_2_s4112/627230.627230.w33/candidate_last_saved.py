import os, time, json
info = {}
try:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal"): info["MemTotal"] = line.strip()
    info["numa_nodes"] = [d for d in os.listdir("/sys/devices/system/node") if d.startswith("node")]
except Exception as e: info["err"]=str(e)
print("DIAG " + json.dumps(info), flush=True)

import numba as nb
import numpy as np
nb.set_num_threads(24)

@nb.njit(parallel=True, fastmath=True, cache=False)
def _sum(b, n):
    s = 0.0
    for i in nb.prange(n):
        s += b[i]
    return s

@nb.njit(parallel=True, fastmath=True, cache=False)
def _copy(a, b, n):
    for i in nb.prange(n):
        a[i] = b[i]

@nb.njit(parallel=True, fastmath=True, cache=False)
def _gather(a, b, ip, n):
    for i in nb.prange(n):
        a[i] = a[i] + b[ip[i]] * 2.0

@nb.njit(fastmath=True, cache=False)
def _sum1(b, n):
    s = 0.0
    for i in range(n):
        s += b[i]
    return s

def t(f, *args, reps=3, mk=None):
    best = 1e30
    for _ in range(reps):
        if mk is not None: args = mk(*args)
        t0 = time.perf_counter(); f(*args); t1 = time.perf_counter()
        best = min(best, t1-t0)
    return best

def s4112(a, b, ip, LEN_1D):
    n = LEN_1D
    GB = 2**30
    a2 = a.copy()
    _gather(a2, b, ip, n)
    g = t(_gather, a2, b, ip, n, mk=lambda a2,b,ip,n: (a.copy(), b, ip, n))
    r = t(_sum, b, n)
    c = t(_copy, a2, b, n, mk=lambda a2,b,n: (a.copy(), b, n))
    r1 = t(_sum1, b, n, reps=1)
    print(f"DIAG N={n} bytes_b={b.nbytes/GB:.3f}GB", flush=True)
    print(f"DIAG gather24 {g*1e3:.2f} ms", flush=True)
    print(f"DIAG read24 {r*1e3:.2f} ms -> {b.nbytes/r/1e9:.1f} GB/s", flush=True)
    print(f"DIAG copy24 {c*1e3:.2f} ms -> {2*b.nbytes/c/1e9:.1f} GB/s(rw)", flush=True)
    print(f"DIAG read1t {r1*1e3:.2f} ms -> {b.nbytes/r1/1e9:.1f} GB/s", flush=True)
    _gather(a, b, ip, n)
    return None
