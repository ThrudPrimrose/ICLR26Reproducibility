"""TSVC tsvc_2 kernel s323 -- vectorized (bit-exact) NumPy implementation.

Reference (in-place):
    for i in range(1, LEN_1D):
        a[i] = b[i - 1] + c[i] * d[i]
        b[i] = a[i] + c[i] * e[i]

Algebra: with u[i] = c[i]*d[i] and v[i] = c[i]*e[i], the recurrence is the
left-to-right scan of the interleaved sequence
    W = [b0, u1, v1, u2, v2, ...]
so that  a[i] = scan[2i-1] and b[i] = scan[2i].  A single np.cumsum of W
reproduces the reference rounding order exactly.
"""
import numpy as np


def s323(a, b, c, d, e, LEN_1D):
    n = int(LEN_1D)
    if n <= 1:
        return None
    m = n - 1
    W = np.empty(2 * m + 1, dtype=b.dtype)
    W[0] = b[0]
    np.multiply(c[1:n], d[1:n], out=W[1::2])
    np.multiply(c[1:n], e[1:n], out=W[2::2])
    np.cumsum(W, out=W)
    a[1:n] = W[1::2]
    b[1:n] = W[2::2]
    return None
