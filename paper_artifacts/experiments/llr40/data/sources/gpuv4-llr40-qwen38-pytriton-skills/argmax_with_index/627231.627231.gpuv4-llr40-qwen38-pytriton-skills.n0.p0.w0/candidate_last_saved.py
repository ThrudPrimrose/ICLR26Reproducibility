import numpy as np, os, time, sys
from concurrent.futures import ThreadPoolExecutor
_pool = ThreadPoolExecutor(max_workers=64)

def run(a, K):
    n = a.shape[0]
    base = n // K
    futs = []
    lo = 0
    for k in range(K):
        hi = n if k == K - 1 else lo + base
        futs.append((lo, _pool.submit(np.argmax, a[lo:hi])))
        lo = hi
    best = None
    bi = 0
    for lo, f in futs:
        i = lo + f.result()
        v = a[i]
        if best is None or v > best:
            best, bi = v, i
    return best, bi

def argmax_with_index(a, out_value, out_index, LEN_1D):
    for K in (20, 24, 28, 32, 36, 40):
        r = 1e18
        for _ in range(3):
            s = time.perf_counter_ns(); run(a, K); e = time.perf_counter_ns()
            r = min(r, e - s)
        print("K=%3d  %8.2f ms" % (K, r / 1e6))
    b, i = run(a, 32)
    out_value[0] = b
    out_index[0] = i
    sys.stdout.flush()
    return None
