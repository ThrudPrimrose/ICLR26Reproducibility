import os, sys, subprocess, time
import numpy as np
import numba
from numba import prange


@numba.njit(parallel=True, fastmath=True)
def _red(x):
    s = 0.0
    for i in prange(x.shape[0]):
        s += x[i]
    return s


@numba.njit(parallel=True, fastmath=True)
def _rw(x):
    for i in prange(x.shape[0]):
        x[i] = x[i] * 1.0000001


@numba.njit(parallel=True, fastmath=True)
def _kernel(a, b, c, d, aa, bb, cc):
    N = aa.shape[0]
    for j in prange(N):
        for i in range(N):
            aa[j, i] += bb[j, i] * cc[j, i]
    for i in prange(N):
        a[i] = b[i] + c[i] * d[i]


def _warm():
    n = 256
    a = np.zeros(n); b = np.ones(n); c = np.ones(n); d = np.ones(n)
    aa = np.zeros((n, n)); bb = np.ones((n, n)); cc = np.ones((n, n))
    _kernel(a, b, c, d, aa, bb, cc)
    _red(bb.reshape(-1))
    _rw(bb.reshape(-1))


_warm()


def _sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout
    except Exception as e:
        return "ERR %r" % e


def s2275(a, b, c, d, aa, bb, cc, LEN_2D):
    L = []
    L.append("gcc: " + _sh("gcc --version 2>&1 | head -1").strip())
    L.append("avx: " + _sh("gcc -march=native -Q --help=target 2>/dev/null | grep -E '\\b(avx512f|avx2|fma)\\b' | head -5").replace("\n", " | ").strip())
    L.append("lscpu: " + _sh("lscpu 2>/dev/null | grep -E 'NUMA|Socket|Model name|L3'").replace("\n", " | ").strip())
    for n in _sh("ls /sys/devices/system/node/ 2>/dev/null | grep node").split():
        try:
            L.append("node %s: %s" % (n, open("/sys/devices/system/node/%s/cpulist" % n).read().strip()))
        except Exception:
            pass
    L.append("affinity: " + str(sorted(os.sched_getaffinity(0))))
    b1 = bb.reshape(-1)
    N2 = b1.shape[0]
    for _ in range(2):
        _red(b1)
    t0 = time.time()
    _red(b1)
    t1 = time.time()
    L.append("read-only 24t: %.2f ms %.0f GB/s" % ((t1 - t0) * 1e3, N2 * 8 / (t1 - t0) / 1e9))
    t0 = time.time()
    _rw(b1)
    t1 = time.time()
    L.append("rw 24t: %.2f ms %.0f GB/s" % ((t1 - t0) * 1e3, N2 * 16 / (t1 - t0) / 1e9))
    sys.stderr.write("\n".join(L) + "\n")
    sys.stderr.flush()
    return None
