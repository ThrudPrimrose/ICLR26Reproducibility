"""Numba-accelerated implementation of the TSVC ``s3111`` kernel.

This kernel computes the sum of all positive elements in the input array ``a``
and stores the result in ``b[0]``.  The reference uses a plain Python loop; a
Numba JIT‑compiled version runs at native speed and matches the ``numba``
baseline used by the benchmark.

The function signature follows the convention employed by the reference NumPy
implementation:

```python
def s3111(a, b, LEN_1D):
    ...
```

* ``a`` – 1‑D ``np.ndarray`` of ``float64`` with logical length ``LEN_1D``.
* ``b`` – 1‑D ``np.ndarray`` of ``float64`` with at least two elements; the sum
  is written to ``b[0]`` (``b[1]`` is unused).
* ``LEN_1D`` – integer specifying how many entries of ``a`` to process.

The implementation uses a helper function compiled with ``@numba.njit`` to
avoid Python overhead.  The outer ``s3111`` wrapper simply forwards the call.
"""

import numpy as np
import numba as nb

@nb.njit(cache=True, fastmath=True)
def _s3111_impl(a, b, LEN_1D):
    """Numba JIT kernel that sums positive values of ``a``.

    Parameters
    ----------
    a : np.ndarray
        Input array (``float64``).
    b : np.ndarray
        Output buffer; ``b[0]`` receives the result.
    LEN_1D : int
        Logical length of ``a`` to process.
    """
    sum_val = 0.0
    # Use a local variable for the limit to avoid repeated attribute look‑ups.
    n = LEN_1D
    for i in range(n):
        if a[i] > 0.0:
            sum_val += a[i]
    b[0] = sum_val
    # No return value – ``None`` signals the in‑place ABI.
    return None

def s3111(a, b, LEN_1D):
    """Public entry point matching the benchmark ABI.

    This thin wrapper forwards to the compiled implementation.  ``a`` and ``b``
    are NumPy arrays provided by the harness; ``LEN_1D`` may be smaller than the
    actual array length, so we slice accordingly inside the JIT function.
    """
    # The JIT function expects ``LEN_1D`` as a plain Python ``int``; ensure the
    # type is correct even if the harness passes a NumPy scalar.
    n = int(LEN_1D)
    # ``_s3111_impl`` works directly on the arrays; slicing is performed in the
    # loop based on ``n``.
    return _s3111_impl(a, b, n)

