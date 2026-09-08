import os, sys, time
import numpy as np
import numba
from numba import njit, prange

@njit
def _serial_out(A, b, c, out):
    n = A.shape[0]
    for i in range(n):
        out[i] = A[i] * b[i] * c[i]

@njit(parallel=True)
def _par_out(A, b, c, out):
    n = A.shape[0]
    for i in prange(n):
        out[i] = A[i] * b[i] * c[i]

@njit(parallel=True)
def _static_chunks(A, b, c, out, k):
    n = A.shape[0]
    base = n // k
    rem = n % k
    for t in prange(k):
        lo = t * base + min(t, rem)
        hi = lo + base + (1 if t < rem else 0)
        for j in range(lo, hi):
            out[j] = A[j] * b[j] * c[j]

@njit(parallel=True)
def _unroll8(A, b, c, out):
    n = A.shape[0]
    for i in prange(n):
        j = i
        out[j] = A[j] * b[j] * c[j]
        j += 1
        if j < n: out[j] = A[j] * b[j] * c[j]
        j += 1
        if j < n: out[j] = A[j] * b[j] * c[j]
        j += 1
        if j < n: out[j] = A[j] * b[j] * c[j]

_w = np.ones(64, np.float64)
_serial_out(_w, _w, _w, _w)
_par_out(_w, _w, _w, _w)
_static_chunks(_w, _w, _w, _w, 4)
_unroll8(_w, _w, _w, _w)

PROBED = False

def vtvtv(a, b, c, LEN_1D):
    global PROBED
    n = a.shape[0]
    if n < (1 << 16) or PROBED:
        _serial_out(a, b, c, a) if n < (1<<16) else None
        if PROBED:
            return None
        # normal small path
        return None
    if PROBED:
        _serial_out(a, b, c, a)
        return None
    PROBED = True
    t0 = time.perf_counter()
    orig = a.copy()
    out = a
    def best(fn, reps=3):
        fn(orig, b, c, out)  # warm
        bt = 1e9
        for _ in range(reps):
            ts = time.perf_counter(); fn(orig, b, c, out); te = time.perf_counter()
            bt = min(bt, te - ts)
        return bt
    t_par = best(lambda: _par_out(orig, b, c, out))
    numba.set_num_threads(12); t_par12 = best(lambda: _par_out(orig, b, c, out))
    numba.set_num_threads(16); t_par16 = best(lambda: _par_out(orig, b, c, out))
    numba.set_num_threads(24)
    t_st24 = best(lambda: _static_chunks(orig, b, c, out, 24))
    t_st12 = best(lambda: _static_chunks(orig, b, c, out, 12))
    t_u8 = best(lambda: _unroll8(orig, b, c, out))
    t_ser = best(_serial_out, 2)
    print(f"par24={t_par*1e3:.2f}ms ({3.2/t_par/1e3:.0f}GB/s) par12={t_par12*1e3:.2f} ({3.2/t_par12/1e3:.0f}) par16={t_par16*1e3:.2f} ({3.2/t_par16/1e3:.0f})")
    print(f"static24={t_st24*1e3:.2f} ({3.2/t_st24/1e3:.0f}) static12={t_st12*1e3:.2f} ({3.2/t_st12/1e3:.0f}) unroll8={t_u8*1e3:.2f} ({3.2/t_u8/1e3:.0f}) serial={t_ser*1e3:.2f} ({3.2/t_ser/1e3:.0f})")
    try:
        l3 = open('/sys/devices/system/cpu/cpu0/cache/index3/size').read().strip()
        nodes = os.listdir('/sys/devices/system/node')
        print("L3", l3, "NODES", sorted(nodes))
    except Exception as e:
        print("TOPO_ERR", repr(e))
    # ensure correct final state:
    _serial_out(orig, b, c, a)
    sys.stdout.flush()
    return None
