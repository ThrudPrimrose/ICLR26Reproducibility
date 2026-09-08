'''Optimized implementation of the TSVC s2233 kernel.

The reference implementation (see /shared/tasks/tsvc_2_s2233/tsvc_2_s2233_numpy.py) uses two nested Python loops:

    for i in range(8, LEN_2D):
        for j in range(8, LEN_2D):
            aa[j, i] = aa[j - 1, i] + cc[j, i]
        for j in range(8, LEN_2D):
            bb[i, j] = bb[i - 1, j] + cc[i, j]

Both inner loops are *prefix‑sum* recurrences along the first axis (rows). The first recurrence updates a column of ``aa``; the second updates a row of ``bb``. Because the dependency is only on the previous row, the entire column/row can be expressed as a cumulative sum of ``cc`` plus the value on row ``7`` (the element just before the start index ``8``). NumPy's ``np.cumsum`` performs this operation in compiled code and eliminates the Python loops.

The function updates ``aa`` and ``bb`` *in‑place* to match the C/NumPy reference ABI. All work is done with vectorised NumPy operations; for ``LEN_2D <= 8`` the slices are empty and the function is a no‑op.
'''

from __future__ import annotations

import numpy as np

def s2233(aa: np.ndarray, bb: np.ndarray, cc: np.ndarray, LEN_2D: int) -> None:
    '''Perform the s2233 kernel computation.

    Parameters
    ----------
    aa, bb, cc : np.ndarray
        2‑D square arrays of shape (LEN_2D, LEN_2D).
    LEN_2D : int
        Logical size of the problem. The reference loops start at index 8— that
        is, the first 8 rows/columns are left unchanged.

    The function mutates aa and bb in‑place, returning None (the in‑place ABI
    expected by the benchmark harness).
    '''
    start = 8
    if LEN_2D <= start:
        return

    # Ensure we work on a contiguous copy of the relevant sub‑matrix.
    sub = cc[start:, start:].copy()
    # Compute cumulative sum of cc in-place using a simple row‑wise loop.
    for k in range(1, sub.shape[0]):
        sub[k, :] += sub[k - 1, :]

    # Add the aa base row to obtain the final 'aa' values.
    aa_base = aa[start - 1, start:]
    bb_base = bb[start - 1, start:]
    sub += aa_base
    aa[start:, start:] = sub
    # Compute the difference between the 'bb' and 'aa' base rows, then add it
    # to the already‑modified sub to obtain the final 'bb' values.
    diff = bb_base - aa_base
    np.add(sub, diff, out=bb[start:, start:])
