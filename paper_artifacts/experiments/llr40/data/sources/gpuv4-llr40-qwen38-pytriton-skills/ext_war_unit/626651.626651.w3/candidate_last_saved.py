import os, sys, time
import numpy as np
import numba
from numba import prange

def _p(*args):
    sys.stdout.write(" ".join(str(x) for x in args) + "\n")
    sys.stdout.flush()

# each kernel writes result into `a` (in-place), reads a and b. n = len.
@numba.njit(parallel=True, cache=False)
def k_copy(a, b, n):           # a[i]=b[i]  16B
    for i in prange(n):
        a[i] = b[i]

@numba.njit(parallel=True, cache=False)
def k_two(a, b, n):            # a[i]=a[i]+b[i] 24B inplace no shift
    for i in prange(n):
        a[i] = a[i] + b[i]

@numba.njit(parallel=True, cache=False)
def k_two_shift(a, b, n):      # needs tmp to be safe -> use out=c? here a[i]=a[i+1]+b[i] UNSAFE in-place; measure with separate out via b as scratch? Instead measure tmp version:
    nn = n - 1
    tmp = np.empty(nn)
    for i in prange(nn):
        tmp[i] = a[i + 1] + b[i]
    for i in prange(nn):
        a[i] = tmp[i]

@numba.njit(parallel=True, cache=False)
def k_two_shift_c(a, b, n):    # c[i]=a[i+1]+b[i] writing into b (24B, shift, no inplace hazard since out=b separate)
    nn = n - 1
    for i in prange(nn):
        b[i] = a[i + 1] + b[i]

@numba.njit(parallel=True, cache=False)
def k_two_plain_c(a, b, n):    # c[i]=a[i]+b[i] into b (24B, no shift)
    for i in prange(n):
        b[i] = a[i] + b[i]

@numba.njit(parallel=True, cache=False)
def k_copy_plain(a, b, n):     # b[i]=a[i] (16B)
    for i in prange(n):
        b[i] = a[i]

def _tm(fn, reps=4):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts)//2]

def ext_war_unit(a, b, LEN_1D):
    n = LEN_1D
    a0 = a.copy(); b0 = b.copy()
    nt = len(os.sched_getaffinity(0))
    numba.set_num_threads(nt)
    # warm
    for f in (k_copy, k_two, k_two_shift, k_two_shift_c, k_two_plain_c, k_copy_plain):
        f(a, b, n)
    a[:] = a0; b[:] = b0
    _p("nt=%d n=%d" % (nt, n))
    def show(name, B, fn):
        a[:] = a0; b[:] = b0
        t = _tm(fn)
        _p("%-16s %6.2f ms  %6.0f GB/s" % (name, t*1e3, B*n/(t/1e3)/1e9))
    show("copy a=b", 16, lambda: k_copy(a, b, n))
    show("copy b=a", 16, lambda: k_copy_plain(a, b, n))
    show("two a=a+b", 24, lambda: k_two(a, b, n))
    show("two b=a+b", 24, lambda: k_two_plain_c(a, b, n))
    show("shift b=a[i+1]+b", 24, lambda: k_two_shift_c(a, b, n))
    show("shift tmp(inplace a)", 40, lambda: k_two_shift(a, b, n))
    # numpy reference for timing
    a[:] = a0; b[:] = b0
    def nprun():
        a[:n-1] = a[1:] + b[:n-1]
    t = _tm(nprun)
    _p("%-16s %6.2f ms  (numpy)" % ("numpy", t*1e3))
    return None

_aa = np.ones(64); _bb = np.ones(64)
for f in (k_copy, k_two, k_two_shift, k_two_shift_c, k_two_plain_c, k_copy_plain):
    f(_aa, _bb, 64)
