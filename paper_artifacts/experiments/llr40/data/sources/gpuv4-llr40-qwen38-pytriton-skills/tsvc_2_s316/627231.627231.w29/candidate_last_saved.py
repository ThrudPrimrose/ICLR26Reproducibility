"""tsvc_2 s316 (fp64): result[0] = min over a[0..LEN_1D-1]."""
import numpy as np


def s316(a, result, LEN_1D):
    info = "a.shape=%s dtype=%s contig=%s result.shape=%s LEN_1D=%s" % (
        tuple(a.shape), a.dtype, bool(a.flags["C_CONTIGUOUS"]),
        tuple(np.asarray(result).shape), LEN_1D)
    print(info, flush=True)
    x = a[0]
    for i in range(1, a.shape[0]):
        if a[i] < x:
            x = a[i]
    result[0] = x
    return None
