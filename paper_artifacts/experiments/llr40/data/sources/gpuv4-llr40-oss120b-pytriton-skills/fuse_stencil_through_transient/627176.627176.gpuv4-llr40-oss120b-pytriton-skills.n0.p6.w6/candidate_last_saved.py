import numpy as np
import numba

@numba.njit(fastmath=True)
def fuse_stencil_through_transient(out, a, LEN_1D):
    n = LEN_1D
    # Compute fused stencil: for i in 1 .. n-3 inclusive
    for i in range(1, n-2):
        s1 = a[i-1] + a[i] + a[i+1]
        s2 = a[i] + a[i+1] + a[i+2]
        out[i] = s1 * s2
    return None
