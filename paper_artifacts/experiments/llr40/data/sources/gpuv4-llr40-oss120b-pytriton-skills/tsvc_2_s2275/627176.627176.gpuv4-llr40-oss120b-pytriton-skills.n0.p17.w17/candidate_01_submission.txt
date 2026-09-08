"""Optimized implementation of the TSVC ``s2275`` kernel using Numba.

The reference implementation (see ``/shared/tasks/tsvc_2_s2275/tsvc_2_s2275_numpy.py``)
updates a 2‑D array ``aa`` with a fused multiply‑add ``aa += bb * cc`` and computes a
vector ``a`` as ``a = b + c * d``.

This version keeps the in‑place ABI (the function returns ``None`` and writes directly to
the output buffers ``a`` and ``aa``).  It combines a NumPy‑based vectorised update for ``a`` with a
Numba‑JITted row‑major loop for ``aa``.  The row‑major ordering improves cache locality
relative to the column‑major order used in the reference loops, yielding a noticeable speed‑up.

The Numba kernel is compiled once (the first time the function is called) and then reused.
Compilation happens inside the timed region for the first repetition, but this cost is small
compared to the overall work for realistic problem sizes.
"""

from __future__ import annotations

import numpy as np
import numba

# Numba kernel that updates ``aa`` in‑place: aa += bb * cc.
# It iterates in row‑major order (j outer, i inner) to maximise memory‑bandwidth utilisation.
# ``fastmath=True`` enables aggressive floating‑point optimisations that are safe for the
# reference's tolerance band.
@numba.njit(parallel=True, fastmath=True)
def _aa_update(aa: np.ndarray, bb: np.ndarray, cc: np.ndarray) -> None:
    N = aa.shape[0]
    for j in numba.prange(N):
        for i in range(N):
            aa[j, i] = aa[j, i] + bb[j, i] * cc[j, i]


def s2275(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray,
          aa: np.ndarray, bb: np.ndarray, cc: np.ndarray, LEN_2D: int) -> None:
    """In‑place version of the ``s2275`` kernel.

    Parameters
    ----------
    a, b, c, d : 1‑D arrays of length ``LEN_2D``
        ``a`` is the output vector; ``b``, ``c`` and ``d`` are inputs.
    aa, bb, cc : 2‑D arrays of shape ``(LEN_2D, LEN_2D)``
        ``aa`` is the output matrix; ``bb`` and ``cc`` are inputs.
    LEN_2D : int
        Logical size of each dimension (not used directly; we rely on the actual array shapes).
    """
    # Compute a = b + c * d using NumPy's vectorised ufuncs.
    np.multiply(c, d, out=a)   # a = c * d
    np.add(a, b, out=a)        # a = b + (c * d)

    # Update aa in‑place with a fused multiply‑add.
    _aa_update(aa, bb, cc)

    # Implicit ``return None`` – the harness detects the in‑place ABI.
    return None

