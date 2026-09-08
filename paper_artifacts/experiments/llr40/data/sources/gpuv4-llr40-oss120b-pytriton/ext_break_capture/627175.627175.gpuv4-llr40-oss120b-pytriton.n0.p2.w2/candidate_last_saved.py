# Optimized implementation of ext_break_capture kernel
# The initialize function is copied from the reference to generate inputs.

from typing import Any, Optional, Tuple
import numpy as np
import numba

def initialize(
    LEN_1D: int,
    K: int,
    datatype: type = np.float64,
    variant_spec: Optional[Any] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Generate inputs for the ext_break_capture kernel.
    The logic follows the reference implementation to ensure identical inputs.
    '''
    if rng is None:
        rng = np.random.default_rng()
    a = rng.uniform(-1000.0, float(K) - 1e-3, LEN_1D).astype(datatype)
    lo_frac, hi_frac = (0.40, 0.60) if int(rng.integers(0, 2)) == 0 else (0.50, 0.70)
    lo = max(0, int(LEN_1D * lo_frac))
    hi = max(lo + 1, int(LEN_1D * hi_frac))
    cut = int(rng.integers(lo, hi)) if LEN_1D > 1 else 0
    a[cut] = datatype(float(K) + 500.0)
    out_index = np.zeros(1, dtype=np.int64)
    out_value = np.zeros(1, dtype=datatype)
    return a, out_index, out_value

@numba.njit
def _ext_break_capture_nb(a, out_index, out_value, LEN_1D, K):
    '''
    Numba-compiled version of ext_break_capture.
    '''
    out_index[0] = -1
    out_value[0] = -1.0
    for i in range(LEN_1D):
        if a[i] > K:
            out_index[0] = i
            out_value[0] = a[i]
            break

# Compile the Numba function at import time to avoid measuring compile overhead.
# Use a tiny dummy array for warm-up.
_dummy_a = np.zeros(1, dtype=np.float64)
_dummy_out_index = np.zeros(1, dtype=np.int64)
_dummy_out_value = np.zeros(1, dtype=np.float64)
_ext_break_capture_nb(_dummy_a, _dummy_out_index, _dummy_out_value, 1, 0)

def ext_break_capture(a, out_index, out_value, LEN_1D, K):
    """
    Wrapper that forwards to the Numba-compiled implementation.
    """
    _ext_break_capture_nb(a, out_index, out_value, LEN_1D, K)
