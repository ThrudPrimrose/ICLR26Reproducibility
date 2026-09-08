import numpy as np
from concurrent.futures import ThreadPoolExecutor

_NTHREADS = 16
_EX = ThreadPoolExecutor(max_workers=64)


def s311(a, sum_out, LEN_1D):
    n = a.size
    if n <= (1 << 18):
        sum_out[0] = a.sum()
        return None
    K = min(_NTHREADS, n >> 18)
    base = n // K
    parts = np.empty(K)
    starts = np.linspace(0, n, K + 1).astype(np.int64)
    chunks = [a[starts[i]:starts[i + 1]] for i in range(K)]
    futs = [_EX.submit(c.sum) for c in chunks]
    for i, f in enumerate(futs):
        parts[i] = f.result()
    sum_out[0] = parts.sum()
    return None
