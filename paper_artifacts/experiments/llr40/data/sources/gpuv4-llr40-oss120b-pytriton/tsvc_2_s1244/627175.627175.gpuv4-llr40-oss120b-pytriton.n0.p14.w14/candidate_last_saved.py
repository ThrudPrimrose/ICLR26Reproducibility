"""Numba-parallel implementation of TSVC kernel ``s1244``.

The reference loop (see ``/shared/tasks/tsvc_2_s1244/tsvc_2_s1244_numpy.py``) is:

```python
for i in range(LEN_1D - 1):
    a[i] = b[i] + c[i] * c[i] + b[i] * b[i] + c[i]
    d[i] = a[i] + a[i + 1]
```

A straightforward NumPy vectorisation (previous submission) was faster than the
pure‑Python reference but still slower than the benchmark baseline, which is a
Numba‑jit compiled version of the same loop.  To beat the baseline we employ a
Numba JIT function with explicit parallelism (``prange``) and fast‑math
optimisations.  The algorithm still follows the exact data‑dependence rules –
``d[i]`` must be computed from the *new* ``a[i]`` and the *original* ``a[i+1]``.
We therefore copy the original ``a`` once (a cheap ``memcpy``) before the
parallel loop.

The JIT compilation is triggered at import time, so the compilation cost is not
included in the timed execution.
"""

from __future__ import annotations

import numpy as np
import numba as nb

# ---------------------------------------------------------------------------
# Numba kernel
# ---------------------------------------------------------------------------

@nb.njit(parallel=True, fastmath=True)
def _s1244_numba(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, LEN_1D: int) -> None:
    """Numba‑parallel implementation of the ``s1244`` kernel.

    Parameters are the same as the reference implementation.
    """
    # Copy the original ``a`` – we only need the tail ``a[1:]`` for the ``d``
    # computation, but copying the whole array is cheap and keeps the code simple.
    a_orig = a.copy()
    # Parallel loop over the logical domain. ``prange`` distributes the work across
    # threads while preserving the semantics because each iteration only reads
    # from ``a_orig`` (which is immutable) and writes to distinct locations of ``a``
    # and ``d``.
    for i in nb.prange(LEN_1D - 1):
        # a[i] = b[i] + c[i]**2 + b[i]**2 + c[i]
        # Using explicit multiplies is marginally faster than ``**2``.
        a_i = b[i]
        c_i = c[i]
        a_i = a_i + c_i * c_i + b[i] * b[i] + c_i
        a[i] = a_i
        d[i] = a_i + a_orig[i + 1]
    # ``a[LEN_1D-1]`` and ``d[LEN_1D-1]`` are left untouched, matching the C reference.
    return None

# ---------------------------------------------------------------------------
# Warm up compilation at import time – this runs once before the benchmark timer
# starts, ensuring the JIT compilation cost is not measured.
# ---------------------------------------------------------------------------

# Create tiny dummy arrays to trigger compilation lazily.  The arrays are of type
# ``float64`` because the benchmark uses the ``fp64`` variant.
_dummy = np.empty(1, dtype=np.float64)
# The call below will compile ``_s1244_numba`` for the signature used by the
# benchmark.  ``LEN_1D`` is set to ``1`` so the loop body is never executed.
_s1244_compiled = _s1244_numba(_dummy, _dummy, _dummy, _dummy, 1)

# ---------------------------------------------------------------------------
# Public API – the harness calls this function directly.
# ---------------------------------------------------------------------------

def s1244(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, LEN_1D: int) -> None:
    """In‑place TSVC ``s1244`` kernel.

    The implementation simply forwards to the JIT‑compiled NumPy/Numba kernel.
    """
    # Guard against degenerate sizes (the reference loop does nothing for LEN_1D <= 1).
    if LEN_1D <= 1:
        return None
    _s1244_numba(a, b, c, d, LEN_1D)
    return None

