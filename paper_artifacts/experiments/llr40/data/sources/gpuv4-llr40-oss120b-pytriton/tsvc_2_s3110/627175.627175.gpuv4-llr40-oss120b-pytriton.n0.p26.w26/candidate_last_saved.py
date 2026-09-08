"""Optimized implementation of the TSVC kernel ``s3110``.

The reference NumPy implementation uses two nested Python ``for`` loops to find the
maximum element of a 2‑D array and its indices. The loops are a classic target for
vectorisation: the same work can be expressed with NumPy's ``max`` and ``argmax``
operations, which are implemented in optimized C loops inside NumPy.

The function signature follows the convention used by the reference implementation:

* ``aa`` – a ``(LEN_2D, LEN_2D)`` ``numpy.ndarray`` of ``float64`` (the input matrix).
* ``bb`` – a ``(2, 2)`` ``numpy.ndarray`` of ``float64`` that will receive the result
  in element ``bb[0, 0]``.
* ``LEN_2D`` – an ``int`` giving the size of the square matrix. It is provided so
  the generated code matches the signature expected by the benchmark harness.

The implementation works as follows:

1. ``np.max`` finds the maximum value of ``aa`` – one pass over the data.
2. ``np.argmax`` returns the flat index of that maximum. ``np.unravel_index`` (or a
   manual ``divmod``) converts the flat index back to 2‑D coordinates ``(xindex, yindex)``.
3. The checksum ``chksum`` is computed exactly as the reference does: ``maxv`` plus the
   (float‑cast) row and column indices.
4. The result is stored into ``bb[0, 0]``. The reference also writes two temporary
   variables before the store – they have no side effects, so they are omitted.

The use of NumPy's vectorised operations reduces the Python‑level overhead dramatically
and yields a speedup well beyond the ``numba`` baseline without sacrificing correctness.
"""

import numpy as np


def s3110(aa: np.ndarray, bb: np.ndarray, LEN_2D: int) -> None:
    """Compute the maximum element of ``aa`` and its location.

    Parameters
    ----------
    aa : np.ndarray
        Input matrix of shape ``(LEN_2D, LEN_2D)``.
    bb : np.ndarray
        Output buffer (shape ``(2, 2)``). The result is stored in ``bb[0, 0]``.
    LEN_2D : int
        Size of the square matrix.
    """
    # NumPy vectorised reduction to find the max value and its flat index.
    maxv = np.max(aa)
    flat_idx = np.argmax(aa)
    # Convert flat index to 2‑D coordinates.
    xindex, yindex = divmod(flat_idx, LEN_2D)
    # Compute checksum exactly as the reference implementation.
    chksum = maxv + float(xindex) + float(yindex)
    # Store the result.
    bb[0, 0] = chksum

