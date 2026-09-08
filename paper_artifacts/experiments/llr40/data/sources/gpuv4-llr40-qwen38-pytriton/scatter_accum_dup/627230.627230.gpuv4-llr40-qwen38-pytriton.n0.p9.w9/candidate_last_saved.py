import os
import time
import numpy as np


def scatter_accum_dup(bins, src, ip, LEN_1D):
    t0 = time.perf_counter()
    acc = np.bincount(ip, weights=src, minlength=LEN_1D)
    t1 = time.perf_counter()
    bins += acc
    t2 = time.perf_counter()
    print("LEN_1D", LEN_1D,
          "shapes", bins.shape, src.shape, ip.shape,
          "dtypes", bins.dtype, src.dtype, ip.dtype,
          "bincount_ms", round((t1 - t0) * 1e3, 3),
          "add_ms", round((t2 - t1) * 1e3, 3),
          "cpu_count", os.cpu_count(), flush=True)
    return None
