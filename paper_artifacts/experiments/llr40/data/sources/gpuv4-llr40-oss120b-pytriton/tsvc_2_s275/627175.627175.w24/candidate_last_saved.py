"""Optimized TSVC ``s275`` kernel using Numba parallel loops.

The reference implementation updates ``aa`` in-place with a column-wise prefix
sum of the element‑wise product of ``bb`` and ``cc``.  A direct Numba JIT with
``prange`` parallelises the outer column loop, providing a substantial speedup on
multi‑core CPUs.
"""

import numpy as np
import numba as nb

@nb.njit(parallel=True, fastmath=True)
def s275(aa, bb, cc, LEN_2D):
    """In‑place TSVC ``s275`` kernel.

    Parameters
    ----------
    aa, bb, cc : np.ndarray
        2‑D ``float64`` arrays of shape ``(LEN_2D, LEN_2D)``.  ``aa`` is updated
        in‑place.
    LEN_2D : int
        The (square) dimension size of the input arrays.
    """
    for i in nb.prange(LEN_2D):
        if aa[0, i] > 0.0:
            for j in range(1, LEN_2D):
                aa[j, i] = aa[j - 1, i] + bb[j, i] * cc[j, i]
