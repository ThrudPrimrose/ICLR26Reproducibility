import numpy as np

def s323(a, b, c, d, e, LEN_1D):
    """Optimized implementation of the TSVC s323 kernel using prefix sum.
    In-place modifies arrays a and b.
    Args:
        a, b, c, d, e: NumPy arrays of length at least LEN_1D.
        LEN_1D: int, the length of the 1D domain.
    """
    # Handle degenerate cases where loop does not execute.
    if LEN_1D <= 1:
        return
    # Compute the incremental term for b: c[i] * (d[i] + e[i]) for i = 1..LEN_1D-1
    # Allocate a temporary buffer for the term.
    tmp = np.empty(LEN_1D - 1, dtype=c.dtype)
    # tmp = d[1:] + e[1:]
    np.add(d[1:LEN_1D], e[1:LEN_1D], out=tmp)
    # tmp = c[1:] * tmp
    np.multiply(c[1:LEN_1D], tmp, out=tmp)
    # Prefix sum on tmp to accumulate contributions for b.
    np.cumsum(tmp, out=tmp)
    # b[0] stays unchanged; b[1:] = b[0] + tmp (broadcast scalar addition).
    b[1:LEN_1D] = tmp + b[0]
    # Compute a[1:] = b[0:LEN_1D-1] + c[1:] * d[1:]
    # Reuse a[1:] as output buffer for the multiplication.
    np.multiply(c[1:LEN_1D], d[1:LEN_1D], out=a[1:LEN_1D])
    a[1:LEN_1D] += b[0:LEN_1D-1]
    # Function returns None (in-place operation).
    return
