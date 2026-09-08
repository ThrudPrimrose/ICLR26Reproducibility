import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

_NUMA = os.sched_getaffinity(0)
_T = max(1, min(32, len(_NUMA)))
if _T > 1:
    _POOL = ThreadPoolExecutor(max_workers=_T)
else:
    _POOL = None
_npmin = np.min


def s316(a, result, LEN_1D):
    n = LEN_1D
    if _POOL is not None and n >= (1 << 20):
        parts = np.array_split(a[:n], _T)
        futs = [_POOL.submit(_npmin, p) for p in parts]
        m = futs[0].result()
        for f in futs[1:]:
            v = f.result()
            if v < m:
                m = v
        result[0] = m
        return None
    result[0] = _npmin(a[:n])
    return None
