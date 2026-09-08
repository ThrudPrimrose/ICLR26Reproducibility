import numpy as np

def s323(a, b, c, d, e, LEN_1D):
    """Vectorized implementation of the TSVC s323 kernel.

    Updates arrays ``a`` and ``b`` in place for ``i = 1 .. LEN_1D-1``.
    The recurrence can be expressed as a simple prefix sum:
      b[i] = b[i-1] + c[i] * (d[i] + e[i])
    and then
      a[i] = b[i-1] + c[i] * d[i]
    """
    if LEN_1D <= 1:
        return
    # Compute the incremental update for b: w[i] = c[i] * (d[i] + e[i])
    w = c * (d + e)
    # Prefix sum of w starting from index 1 (w[0] is unused)
    # b[0] stays unchanged; for i>=1: b[i] = b[0] + sum_{k=1..i} w[k]
    b[1:LEN_1D] = b[0] + np.cumsum(w[1:LEN_1D])
    # Now compute a using the newly updated b values
    a[1:LEN_1D] = b[0:LEN_1D-1] + c[1:LEN_1D] * d[1:LEN_1D]
