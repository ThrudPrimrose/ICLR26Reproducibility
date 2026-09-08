import os
import numpy as np

def s3110(aa, bb, LEN_2D):
    print("CPUS os.cpu_count:", os.cpu_count())
    try:
        print("AFFINITY:", sorted(os.sched_getaffinity(0)))
    except Exception as e:
        print("affinity err", e)
    print("numpy", np.__version__)
    print("aa shape", aa.shape, "dtype", aa.dtype, "bb", bb.shape, bb.dtype, "LEN_2D", LEN_2D)
    try:
        import numba
        print("numba", numba.__version__)
        print("numba default threads", numba.config.NUMBA_NUM_THREADS)
    except Exception as e:
        print("numba err", e)
    try:
        import triton
        print("triton", triton.__version__)
    except Exception as e:
        print("triton err", e)
    import sys
    sys.stdout.flush()
    flat = aa.argmax()
    i, j = divmod(int(flat), int(LEN_2D))
    bb[0, 0] = aa.flat[flat] + float(i) + float(j)
    return None
