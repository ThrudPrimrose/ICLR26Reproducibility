import numpy as np
import time
import numba as nb
from numba import prange


@nb.njit(parallel=True, fastmath=False, boundscheck=False)
def _core(a, b, c, aa, bb, n, T):
    nt = (n + T - 1) // T
    for t in prange(nt):
        i0 = t * T
        i1 = i0 + T
        if i1 > n:
            i1 = n
        for i in range(i0, i1):
            a[i] = a[i] + b[i] * c[i]
        for j in range(1, n):
            for i in range(i0, i1):
                aa[j, i] = aa[j - 1, i] + bb[j, i] * a[i]


@nb.njit(parallel=True, fastmath=False, boundscheck=False)
def _rw(x, y, m):
    for i in prange(m):
        y[i] = x[i]


def _timeit(f, reps=3):
    best = 1e30
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best


def s235(a, b, c, aa, bb, LEN_2D):
    n = LEN_2D
    print("DIAG n=%d nbthreads=%s start" % (n, getattr(nb.config, "NUMBA_NUM_THREADS", "?")), flush=True)
    rows = min(n, 1500)
    x = bb[:rows].reshape(-1)
    y = np.zeros(rows * n)
    t_rw = _timeit(lambda: _rw(x, y, x.size), reps=2)
    print("DIAG rw r+w=%d bytes t_ms=%.2f -> %.1f GB/s" % (2 * x.nbytes, t_rw * 1000, 2 * x.nbytes / t_rw / 1e9), flush=True)
    for T in (384, 427, 449, 512, 570, 640, 853):
        tt = _timeit(lambda: _core(a, b, c, aa, bb, n, T), reps=3)
        print("DIAG T=%-5d nt=%-3d t_ms=%.2f eff2u_gb_s=%.1f" % (T, (n + T - 1) // T, tt * 1000, (2.0 * aa.nbytes) / tt / 1e9), flush=True)
    _core(a, b, c, aa, bb, n, 512)
    return None


def _warm():
    n = 65
    a = np.zeros(n); b = np.ones(n); c = np.ones(n)
    aa = np.zeros((n, n)); bb = np.ones((n, n))
    x = bb.reshape(-1)
    y = np.zeros(x.size)
    _core(a, b, c, aa, bb, n, 32)
    _rw(x, y, x.size)


_warm()
