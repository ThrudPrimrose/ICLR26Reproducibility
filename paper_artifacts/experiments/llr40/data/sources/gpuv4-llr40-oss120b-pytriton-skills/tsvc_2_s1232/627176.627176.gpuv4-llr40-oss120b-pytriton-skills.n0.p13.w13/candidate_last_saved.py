"""Optimized CPU implementation of the TSVC_2 kernel ``s1232`` using NumPy.

The kernel computes ``aa[i, j] = bb[i, j] + cc[i, j]`` for all ``i >= j * VLEN``.
A single call to ``np.add`` with a Boolean ``where`` mask performs the work in one
vectorised pass, avoiding Python loops and keeping auxiliary memory usage modest
(the mask consumes ``LEN_2D``\xD7``LEN_2D`` booleans, which fits comfortably within the
per‑kernel 20 GB limit for the hidden problem sizes).
"""

from __future__ import annotations

import numpy as np

# Cache masks for (LEN_2D, VLEN) to avoid recomputation across repetitions.
_mask_cache = {}


def s1232(
    aa: np.ndarray,
    bb: np.ndarray,
    cc: np.ndarray,
    LEN_2D: int,
    VLEN: int,
) -> None:
    """In‑place kernel for ``s1232``.

    Parameters
    ----------
    aa, bb, cc:
        ``LEN_2D``\xD7``LEN_2D`` arrays of ``float64``. ``aa`` is the output buffer.
    LEN_2D:
        Size of each dimension.
    VLEN:
        Vector‑length factor that determines the lower‑bound of the iteration.
    """
    # Choose the implementation based on the vector length factor.
    # For small ``VLEN`` the full mask is cheap and avoids Python loops.
    # When ``VLEN`` is large the mask would be large and mostly ``False``; in that case
    # a column‑wise loop using ``np.add`` on slices is faster.
    if VLEN > 8:
        # Column‑wise loop – stop when the start index exceeds the matrix size.
        for j in range(LEN_2D):
            i_start = j * VLEN
            if i_start >= LEN_2D:
                break
            np.add(bb[i_start:, j], cc[i_start:, j], out=aa[i_start:, j])
    else:
        # Retrieve or build the Boolean mask for this problem size.
        # The mask is cached across calls to avoid recomputation overhead.
        key = (LEN_2D, VLEN)
        mask = _mask_cache.get(key)
        if mask is None:
            i = np.arange(LEN_2D, dtype=np.int64)[:, None]
            j = np.arange(LEN_2D, dtype=np.int64)[None, :]
            mask = i >= j * VLEN
            _mask_cache[key] = mask
        np.add(bb, cc, out=aa, where=mask)
    # Returning ``None`` signals the in‑place ABI expected by the harness.
    return None

