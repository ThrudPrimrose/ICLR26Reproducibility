'''Optimized implementation for the tsvc_2_s235 kernel.

The kernel updates ``a`` and ``aa`` as follows (0-based indexing):
    a[i] = a[i] + b[i] * c[i]
    for j in 1..LEN_2D-1:
        aa[j, i] = aa[j-1, i] + bb[j, i] * a[i]

Two strategies are used based on the memory layout of the 2-D input arrays:

* **C\-contiguous** (row\-major) arrays: a row\-wise loop with a reusable
  temporary buffer to minimise allocations.
* **Fortran\-contiguous** (column\-major) arrays: a fully vectorised approach
  using ``np.multiply`` and ``np.cumsum`` along the rows (axis 0).  The
  operations are performed in\-place and require only a small ``row0`` buffer.

Both ``a`` and ``aa`` are updated in\-place; the function returns ``None`` to match
the in\-place C ABI expected by the harness.
''' 
import numpy as np

# Global reusable buffer for the C\-order path.
_row_buf = None

def s235(a, b, c, aa, bb, LEN_2D):
    global _row_buf
    # Update ``a`` in\-place.
    a += b * c
    # Choose algorithm based on layout.
    if aa.flags['F_CONTIGUOUS'] or bb.flags['F_CONTIGUOUS']:
        # --- Fortran\-order (column\-major) path ---
        # Scale ``bb`` by ``a`` in\-place.
        np.multiply(bb, a, out=bb)
        # Preserve the first scaled row for later subtraction.
        row0 = bb[0, :].copy()
        # Cumulative sum down the rows (axis=0) – contiguous for Fortran layout.
        np.cumsum(bb, axis=0, out=bb)
        # Subtract the contribution of the first row from the subsequent rows.
        bb[1:, :] -= row0
        # Update ``aa``: the first row stays unchanged.
        aa[1:, :] = aa[0, :] + bb[1:, :]
    else:
        # --- C\-order (row\-major) path ---
        # Ensure a reusable buffer of the appropriate size exists.
        if _row_buf is None or _row_buf.shape[0] != LEN_2D:
            _row_buf = np.empty(LEN_2D, dtype=a.dtype)
        # Row\-wise update using the buffer to avoid extra temporaries.
        for j in range(1, LEN_2D):
            np.multiply(bb[j, :], a, out=_row_buf)
            np.add(aa[j-1, :], _row_buf, out=aa[j, :])
    return None
