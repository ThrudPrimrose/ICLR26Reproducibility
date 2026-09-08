import numpy as np
import os, sys

def s3111(a, b, LEN_1D):
    b[0] = np.sum(a[a > 0.0])
    try:
        aff = len(os.sched_getaffinity(0))
        import numpy.version
        print("PROBE shape=", a.shape, "dtype=", a.dtype, "nbytes=", a.nbytes,
              "LEN_1D=", LEN_1D, "b.shape=", b.shape, "b.dtype=", b.dtype,
              "cpu_count=", os.cpu_count(), "affinity=", aff, file=sys.stdout)
        try:
            import numpy.core._multiarray_umath as _m
            print("PROBE openblas:", getattr(_m, "__config__", None) is not None, file=sys.stdout)
        except Exception as e:
            print("PROBE ob err", e, file=sys.stdout)
        sys.stdout.flush()
    except Exception as e:
        print("PROBE-ERR", e, file=sys.stdout)
        sys.stdout.flush()
