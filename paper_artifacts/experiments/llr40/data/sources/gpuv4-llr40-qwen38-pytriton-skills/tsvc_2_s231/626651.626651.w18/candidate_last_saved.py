import numpy as np, os, time
import numba
from numba import njit, prange

@njit(parallel=True, nogil=True)
def _par(aa, bb, n):
    for i in prange(n):
        acc = aa[0, i]
        for j in range(1, n):
            acc += bb[j, i]
            aa[j, i] = acc

def _warm():
    a = np.zeros((8, 8)); b = np.zeros((8, 8))
    _par(a, b, 8)
_warm()

def s231(aa, bb, LEN_2D):
    import numba.np.ufunc as _uf
    try:
        nt = numba.get_num_threads()
    except Exception as e:
        nt = str(e)
    print("ENV cpu_count=%s affinity=%s numba_nt=%s OMP=%r" % (
        os.cpu_count(), len(os.sched_getaffinity(0)), nt,
        os.environ.get("OMP_NUM_THREADS")), flush=True)
    t0 = time.perf_counter()
    if LEN_2D > 1:
        _par(aa, bb, LEN_2D)
    t1 = time.perf_counter()
    print("PAR_MS=%.3f" % ((t1 - t0) * 1e3), flush=True)
    return None
