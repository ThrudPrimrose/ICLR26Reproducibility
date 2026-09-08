import numpy as np

def s311(a, sum_out, LEN_1D):
    """Compute the sum of input array `a` of length `LEN_1D`.
    Stores result in `sum_out[0]` in-place.
    """
    # Use NumPy's fast sum implementation.
    # The slice ensures we sum only the first LEN_1D elements if a is larger.
    sum_out[0] = np.sum(a[:LEN_1D])
    # In-place version: return None
    return None
