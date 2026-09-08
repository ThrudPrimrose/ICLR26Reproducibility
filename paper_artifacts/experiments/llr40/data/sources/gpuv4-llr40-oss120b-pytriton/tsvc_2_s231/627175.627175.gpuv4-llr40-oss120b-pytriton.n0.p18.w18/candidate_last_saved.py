"""Optimized implementation of the TSVC ``s231`` kernel for the
``loop_level_reasoning/tsvc_2_s231/tsvc_2_s231`` benchmark.

The reference implementation (see ``/shared/tasks/tsvc_2_s231/tsvc_2_s231_numpy.py``)
uses two explicit Python ``for`` loops:

```python
for i in range(LEN_2D):
    for j in range(1, LEN_2D):
        aa[j, i] = aa[j - 1, i] + bb[j, i]
```

For each column ``i`` this computes a cumulative sum of ``bb`` added to the
original ``aa[0, i]`` value. The naïve loops are very slow for large
``LEN_2D`` because they execute Python byte‑code for every element and they
perform non‑contiguous memory accesses (the inner loop strides over the first
dimension).

We replace the nested loops with a single NumPy ``cumsum`` call which is
implemented in compiled C code and therefore runs at native speed. The
algorithm is:

* ``bb[1:, :]`` contains all rows except the first.
* ``np.cumsum(bb[1:, :], axis=0)`` returns the cumulative sum of those rows
  *per column*.
* Adding ``aa[0, :]`` (broadcasted across rows) yields the exact values that
  the reference kernel would write into ``aa[1:, :]``.

The first row of ``aa`` must remain unchanged, matching the reference
behaviour.

The function mutates ``aa`` in‑place and returns ``None`` – this matches the
expected C‑style “in‑place” ABI used by the benchmark harness.
"""

from __future__ import annotations

import numpy as np
import numba

@numba.njit(parallel=True, fastmath=True)
def _s231_numba(aa, bb, LEN_2D):
    for i in numba.prange(LEN_2D):
        for j in range(1, LEN_2D):
            aa[j, i] = aa[j-1, i] + bb[j, i]

def s231(aa: np.ndarray, bb: np.ndarray, LEN_2D: int) -> None:
    """In‑place compute ``aa[j, i] = aa[j‑1, i] + bb[j, i]`` for ``j >= 1``.

    Parameters
    ----------
    aa : np.ndarray of shape (LEN_2D, LEN_2D) and dtype float64
        Input/output array. The first row (index 0) is left untouched; the
        remaining rows are overwritten with the cumulative‑sum result.
    bb : np.ndarray of shape (LEN_2D, LEN_2D) and dtype float64
        RHS operand.
    LEN_2D : int
        Dimension size.
    """
    _s231_numba(aa, bb, LEN_2D)

