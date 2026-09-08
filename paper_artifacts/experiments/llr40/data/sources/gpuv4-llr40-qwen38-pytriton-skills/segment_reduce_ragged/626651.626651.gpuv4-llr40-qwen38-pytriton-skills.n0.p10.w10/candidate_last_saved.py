import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

try:
    _AFF = len(os.sched_getaffinity(0))
except Exception:
    _AFF = os.cpu_count() or 1

_MAXT = 24
_PER_T = 2_500_000
_pool = None
def _pool_of():
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(max_workers=_MAXT)
    return _pool

def _chunk(rp, val, w, out, s0, s1):
    e0 = rp[s0]; e1 = rp[s1]
    vw = val[e0:e1] * w[e0:e1]
    lens = rp[s0 + 1:s1 + 1] - rp[s0:s1]
    ne = np.nonzero(lens > 0)[0]
    out[s0:s1] = 0.0
    if ne.size == 0:
        return
    starts = rp[s0 + ne]
    last_end = rp[s0 + ne[-1] + 1]
    res = np.add.reduceat(vw[:last_end - e0], starts - e0)
    out[s0 + ne] = res

def _warm():
    rp = np.array([0, 0, 2, 5], np.int64)
    val = np.array([1., 2, 3, 4, 5.])
    w = np.array([1., 1, 1, 1, 1.])
    out = np.zeros(3)
    _chunk(rp, val, w, out, 0, 3)
_warm()

def segment_reduce_ragged(row_ptr, val, w, out, NSEG):
    if NSEG <= 0:
        return None
    total = val.shape[0]
    T = (total + _PER_T - 1) // _PER_T
    if T > _MAXT: T = _MAXT
    if T > _AFF: T = _AFF
    if T < 1: T = 1
    if T == 1:
        _chunk(row_ptr, val, w, out, 0, NSEG)
    else:
        c = (np.arange(1, T) * total) // T
        sb = np.searchsorted(row_ptr, c, side='right')
        sb = np.concatenate((np.array([0], np.int64), sb, np.array([NSEG], np.int64)))
        ex = _pool_of()
        futs = [ex.submit(_chunk, row_ptr, val, w, out, int(sb[i]), int(sb[i + 1])) for i in range(T)]
        for f in futs:
            f.result()
    return None
