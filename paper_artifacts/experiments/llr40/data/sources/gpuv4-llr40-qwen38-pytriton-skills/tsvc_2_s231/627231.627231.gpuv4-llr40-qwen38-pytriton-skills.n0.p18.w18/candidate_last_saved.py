import os, time
import numpy as np
import numba as nb

@nb.njit(cache=False)
def _worker(aa, bb, N, nthreads):
    C = (N + nthreads - 1) // nthreads
    for t in nb.prange(nthreads):
        i0 = t * C
        i1 = N if i0 + C > N else i0 + C
        L = i1 - i0
        for j in range(1, N):
            prev = aa[j-1, i0:i1]
            cur = aa[j, i0:i1]
            add = bb[j, i0:i1]
            for k in range(L):
                cur[k] = prev[k] + add[k]

@nb.njit(cache=False)
def _stream1(aa, bb, N):
    for j in range(N):
        cur = aa[j]; add = bb[j]
        for k in range(N):
            cur[k] = cur[k] + add[k]

def _t(fn, reps=3):
    best = 1e30
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter()-t0)
    return best

def _probe():
    N = 11000
    lines = []
    a = np.random.uniform(-1000, 1000, size=(N, N))
    b = np.random.uniform(-1000, 1000, size=(N, N))
    c = np.empty_like(a)
    gb = N*N*8/1e9
    t = _t(lambda: c[:].__setitem__(slice(None), a)); lines.append(f"memcpy 968MB: {t*1e3:.1f}ms {gb/t:.0f}GB/s")
    t = _t(lambda: np.add(a, b, out=c)); lines.append(f"np.add 2.9GB: {t*1e3:.1f}ms {3*gb/t:.0f}GB/s")
    nb.set_num_threads(1)
    t = _t(lambda: _stream1(c.copy(), b, N), reps=2); lines.append(f"numba stream 1thr: {t*1e3:.1f}ms {3*gb/t:.0f}GB/s")
    for nt in (8, 16, 24):
        nb.set_num_threads(nt)
        t = _t(lambda: _worker(c.copy(), b, N, nt), reps=2); lines.append(f"worker nt={nt}: {t*1e3:.1f}ms {2*gb/t:.0f}GB/s (traffic)")
    print("PROBE " + " | ".join(lines), flush=True)

def s231(aa, bb, LEN_2D):
    for i in range(LEN_2D):
        for j in range(1, LEN_2D):
            aa[j, i] = aa[j-1, i] + bb[j, i]

_probe()
