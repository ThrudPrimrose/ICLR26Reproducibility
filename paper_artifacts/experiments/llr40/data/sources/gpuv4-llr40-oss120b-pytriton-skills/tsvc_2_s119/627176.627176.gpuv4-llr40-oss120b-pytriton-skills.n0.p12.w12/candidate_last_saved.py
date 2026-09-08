"""Optimized implementation of the TSVC kernel ``s119``.

The reference (NumPy) implementation uses two nested Python loops:

    for i in range(1, LEN_2D):
        for j in range(1, LEN_2D):
            aa[i, j] = aa[i - 1, j - 1] + bb[i, j]

The inner loop is a pure element‑wise addition that can be expressed with a NumPy
ufunc.  The outer loop cannot be parallelised because each row ``i`` depends on the
already‑updated values of row ``i‑1`` (the ``aa[i-1, j-1]`` term).  However, we can
still eliminate the inner Python loop entirely by performing a vectorised slice
addition for each row.  Using ``np.add(..., out=...)`` avoids allocating a temporary
array for the sum, reducing memory traffic.

The function follows the *in‑place* ABI expected by the harness: it mutates the
``aa`` buffer directly and returns ``None``.  The ``LEN_2D`` argument is retained
for compatibility with the reference signature.
"""

import numpy as np


def s119(aa: np.ndarray, bb: np.ndarray, LEN_2D: int) -> None:
    """Update ``aa`` in place according to the TSVC ``s119`` kernel.

    Parameters
    ----------
    aa : np.ndarray
        A square 2‑D array of shape ``(LEN_2D, LEN_2D)``.  It is both an input and the
        output of the kernel.  The first row and first column are left untouched
        (the reference kernel starts the loops at index ``1``).
    bb : np.ndarray
        A square 2‑D array of the same shape as ``aa``.
    LEN_2D : int
        The dimension size of the square arrays.

    Notes
    -----
    The implementation iterates over the rows ``i = 1 … LEN_2D-1`` and performs a
    vectorised addition of the slice ``aa[i-1, :-1]`` (the previously updated row) and
    ``bb[i, 1:]`` into ``aa[i, 1:]``.  ``np.add`` is used with the ``out`` argument to
    write the result directly into ``aa`` without creating an intermediate temporary.
    """
    # Iterate over rows, vectorising the inner loop with a NumPy ufunc.
    for i in range(1, LEN_2D):
        # Compute aa[i, 1:] = aa[i-1, :-1] + bb[i, 1:]
        np.add(aa[i-1, :-1], bb[i, 1:], out=aa[i, 1:])

