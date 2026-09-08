"""Optimized implementation of TSVC tsvc_2 kernel s252.

Reference semantics:
    t = 0.0
    for i in range(LEN_1D):
        s = b[i] * c[i]
        a[i] = s + t
        t = s

This is a shifted accumulation: a[i] = s[i] + s[i-1] with s = b*c and s[-1] = 0.
Vectorized with two ufunc passes; the shifted add reads a's own previous
elements (t == s[i-1] already stored in a[i-1]).
"""

import numpy as np


def s252(a, b, c, LEN_1D):
    n = LEN_1D
    if n <= 0:
        return None
    np.multiply(b[:n], c[:n], out=a[:n])
    if n > 1:
        a[1:n] += a[:n - 1]
    return None
