"""Optimized Python implementation for the `quasi_affine_reduce_odd` kernel.

The reference reference (see `/shared/tasks/quasi_affine_reduce_odd/quasi_affine_reduce_odd_numpy.py`)
implements the kernel as a Python loop:

    out[0] = 0.0
    for i in range(1, LEN_1D, 2):
        out[0] = out[0] + a[i]

That approach incurs a massive Python overhead because the loop runs in the interpreter. The benchmark measures the speedup over the baseline NumPy/Numba reference, so we need a vectorized solution that performs the reduction in native NumPy code.

The kernel computes the sum of the elements of ``a`` at odd indices (starting at 1) up to
``LEN_1D`` (exclusive). The input ``a`` is a one-dimensional ``np.ndarray`` of ``float64`` and
``out`` is a one-element ``np.ndarray`` of the same dtype. ``LEN_1D`` may be smaller than
``a.shape[0]``; we must respect that bound.

We implement the in-place ABI: the function writes the result into ``out[0]`` and returns
``None``. This avoids allocating a new output array, which the harness would otherwise need to copy back to the caller.

The implementation uses a single NumPy slice and ``np.sum`` (or the ``.sum()`` method) which
executes the reduction in compiled C code, giving a speedup of several hundred× over the
reference loop. The slice ``a[1:LEN_1D:2]`` creates a view (no copy), so the memory traffic is
identical to the original algorithm: each odd element is read exactly once.
"""

from __future__ import annotations

import numpy as np

def quasi_affine_reduce_odd(a: np.ndarray, out: np.ndarray, LEN_1D: int) -> None:
    """Sum the odd-indexed elements of ``a`` up to ``LEN_1D``.

    Parameters
    ----------
    a : np.ndarray
        Input array of shape ``(LEN_1D,)`` (or larger). Must be ``float64``.
    out : np.ndarray
        Output array of shape ``(1,)`` where the sum will be stored.
    LEN_1D : int
        Logical length of ``a`` to consider.
    """
    # Use a NumPy slice to select odd indices: start at 1, step by 2.
    # The slice creates a view, so no additional memory is allocated.
    # np.sum runs in compiled code and writes the result directly into out[0].
    # Using the ``out`` parameter of np.sum avoids creating a temporary scalar.
    # However, out must be an array of the same shape as the result; out is (1,).
    # ``np.sum(..., out=out)`` writes the scalar into out[0] in place.
    # For clarity and compatibility we compute the sum and assign.
    out[0] = a[1:LEN_1D:2].sum()
    # No return value: the in-place ABI expects None.
    return None

