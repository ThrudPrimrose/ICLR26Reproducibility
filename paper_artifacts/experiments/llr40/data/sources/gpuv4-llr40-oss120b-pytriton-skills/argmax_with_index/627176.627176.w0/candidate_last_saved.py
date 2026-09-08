"""Optimized implementation of the ``argmax_with_index`` kernel for the
+``loop_level_reasoning/argmax_with_index/argmax_with_index`` benchmark.
+
+The reference implementation (see ``/shared/tasks/argmax_with_index``) uses a
+Python ``for`` loop to scan the input array ``a`` and compute the maximum value
+and its index.  The baseline used by the harness is a Numba ``@njit`` version of
+the same loop, which is already compiled to native code but still performs a
+scalar scan.
+
+NumPy provides highly‑optimized C loops for reduction operations.  ``np.argmax``
+returns the *first* index of the maximum element, matching the reference's
+behaviour (the reference updates the index only when ``a[i] > x``; it never
+updates on equality, so the first occurrence wins).  The value can then be read
+directly from ``a`` using that index.
+
+Both the *functional* and *in‑place* ABI are supported.  The harness may either:
+
+* pass ``out_value`` and ``out_index`` as pre‑allocated ``np.ndarray`` objects of
+  shape ``(1,)`` and expect the function to write the results into them and
+  return ``None`` (the C‑style in‑place convention), or
+* expect the function to return the outputs.  Writing in‑place is cheap because
+  it avoids an extra allocation.
+
+The implementation therefore writes directly into the provided output buffers
+and returns ``None``.  It relies only on NumPy, which is imported at module load
+time (outside the timed region).
+"""

from __future__ import annotations

import numpy as np


def argmax_with_index(a: np.ndarray, out_value: np.ndarray, out_index: np.ndarray, LEN_1D: int) -> None:
    """Find the maximum element of ``a`` and its index.

    Parameters
    ----------
    a : np.ndarray
        Input 1‑D array of length ``LEN_1D``.  Its dtype is typically ``float64``.
    out_value : np.ndarray
        Output buffer of shape ``(1,)`` where the maximum value will be stored.
    out_index : np.ndarray
        Output buffer of shape ``(1,)`` where the index of the maximum will be
        stored (as a Python ``int``/NumPy integer).  The reference uses a 64‑bit
        integer type.
    LEN_1D : int
        Length of ``a``.  The argument is kept for API compatibility; the
        implementation does not need it because ``a.shape[0]`` provides the same
        information.

    Returns
    -------
    None
        Results are written in‑place to ``out_value`` and ``out_index``.
    """

    # ``np.argmax`` scans the array in C and returns the first occurrence of the
    # maximum value, matching the reference semantics.
    idx = int(np.argmax(a))
    # ``a[idx]`` fetches the maximum value; this is a single load and does not
    # trigger a second full scan.
    max_val = a[idx]

    # Write results back to the caller‑provided buffers.  ``out_index`` is often
    # an ``int64`` array — NumPy will cast ``idx`` appropriately.
    out_index[0] = idx
    out_value[0] = max_val

    # In‑place ABI: the function returns ``None``.
    return None
