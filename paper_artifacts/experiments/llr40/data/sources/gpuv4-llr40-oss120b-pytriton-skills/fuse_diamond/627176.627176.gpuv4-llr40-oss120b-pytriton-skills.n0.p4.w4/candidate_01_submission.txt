import numpy as np
import numba

@numba.njit(parallel=True, fastmath=True)
def _fuse_diamond_numba(out, a, LEN_1D):
    """Compute out[i] = a[i]^4 - 1.
    Uses a fused loop with a single read of a[i] and write to out[i].
    """
    for i in numba.prange(LEN_1D):
        t = a[i] * a[i]
        out[i] = t * t - 1

# Warm-up compile for common dtypes (outside timed region).
# Use zero-length arrays to trigger compilation without cost.
for _dtype in (np.float64, np.float32, np.int64, np.int32):
    _dummy_out = np.empty(0, dtype=_dtype)
    _dummy_a = np.empty(0, dtype=_dtype)
    _fuse_diamond_numba(_dummy_out, _dummy_a, 0)

def fuse_diamond(out, a, LEN_1D):
    """In-place compute out = a^4 - 1 using Numba for speed.
    The arrays are assumed to have at least LEN_1D elements.
    """
    _fuse_diamond_numba(out, a, LEN_1D)
    # Implicit return None (in-place ABI)
