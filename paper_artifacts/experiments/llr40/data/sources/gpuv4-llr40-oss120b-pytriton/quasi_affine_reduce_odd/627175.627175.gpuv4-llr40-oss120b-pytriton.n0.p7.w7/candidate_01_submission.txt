import numpy as np

def quasi_affine_reduce_odd(a, out, LEN_1D):
    """Sum a[i] for i in range(1, LEN_1D, 2) and store into out[0]."""
    # Use NumPy slicing for vectorized sum; ensure dtype is preserved.
    # The slice respects LEN_1D, which may be less than len(a).
    out[0] = np.sum(a[1:LEN_1D:2], dtype=a.dtype)
    # Nothing to return (in-place).
    return None
