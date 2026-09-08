"""Optimized odd-index reduction: out[0] = sum(a[1:LEN_1D:2])."""
import numpy as np

def quasi_affine_reduce_odd(a, out, LEN_1D):
    out[0] = np.sum(a[1:LEN_1D:2])
    return None
