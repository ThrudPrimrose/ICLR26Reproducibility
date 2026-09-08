'''Optimized Python implementation of the ``tsvc_2`` kernel ``s1232`` using a NumPy ``where`` mask.

The reference implementation uses a double Python loop which is slow.  Here we build a boolean
mask that encodes the condition ``i >= j * VLEN`` and then perform the addition in a single
NumPy ``add`` call with ``where=mask``.  The mask is created on each call to avoid retaining a
large number of masks when the kernel is invoked repeatedly with different ``VLEN`` values.

The function updates ``aa`` in place and returns ``None`` to match the C‑style ABI expected by
the benchmark harness.
''' 

from __future__ import annotations

import numpy as np


def s1232(aa: np.ndarray, bb: np.ndarray, cc: np.ndarray, LEN_2D: int, VLEN: int) -> None:
    """In-place compute ``aa[i, j] = bb[i, j] + cc[i, j]`` for ``i >= j * VLEN``.

    Parameters
    ----------
    aa, bb, cc:
        2-D ``float64`` (or compatible) arrays of shape ``(LEN_2D, LEN_2D)``. ``aa`` is
        overwritten with the result.
    LEN_2D:
        The dimension of the square matrices.
    VLEN:
        Vector length controlling the start index of each column.
    """
    if LEN_2D == 0:
        return
    # Build the mask for the current dimensions and VLEN.
    i = np.arange(LEN_2D)[:, None]
    j = np.arange(LEN_2D)[None, :]
    mask = i >= j * VLEN
    # Perform the addition only where the mask is true; ``aa`` retains its original values elsewhere.
    np.add(bb, cc, out=aa, where=mask)
    # No explicit return – ``aa`` now holds the result.

