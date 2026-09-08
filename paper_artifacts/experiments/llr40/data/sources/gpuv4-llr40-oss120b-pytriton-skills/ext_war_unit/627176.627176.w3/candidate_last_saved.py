import numpy as np
import numba

# Numba JIT kernel performing the in-place update.
@numba.njit
def ext_war_unit(a, b, LEN_1D):
    """Loop: a[i] = a[i + 1] + b[i] for i in 0..LEN_1D-2.
    This function modifies ``a`` in place and returns ``None`` to follow the
    in‑place C ABI convention.
    """
    for i in range(LEN_1D - 1):
        a[i] = a[i + 1] + b[i]

# Warm up the JIT compilation for the expected dtype (float64).
_dummy_a = np.empty(2, dtype=np.float64)
_dummy_b = np.empty(2, dtype=np.float64)
ext_war_unit(_dummy_a, _dummy_b, 2)
