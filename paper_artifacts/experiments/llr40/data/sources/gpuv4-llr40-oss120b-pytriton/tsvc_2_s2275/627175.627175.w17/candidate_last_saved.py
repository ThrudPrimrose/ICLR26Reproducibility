'''Optimized TSVC s2275 kernel using in-place NumPy operations.'''
from __future__ import annotations

import numpy as np

def s2275(a, b, c, d, aa, bb, cc, LEN_2D):
    np.multiply(c, d, out=c)  # c now = c * d
    np.add(b, c, out=a)       # a = b + c
    np.multiply(bb, cc, out=bb)  # bb now = bb * cc
    np.add(aa, bb, out=aa)       # aa = aa + bb
