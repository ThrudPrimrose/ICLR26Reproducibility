"""Optimized implementation of the ``scatter_accum_dup`` benchmark.

The reference kernel performs an indexed accumulate of ``src`` into ``bins`` using the
index vector ``ip``:

    for i in range(LEN_1D):
        bins[ip[i]] = bins[ip[i]] + src[i]

``ip`` may contain duplicate entries, which turns the operation into a reduction.  A
naïve Python loop is orders of magnitude slower than the reference NumPy–JIT baseline.
We accelerate the operation by delegating the accumulation to a Numba JIT‑compiled
loop, which runs at native C speed and correctly handles duplicate indices.

The function follows the *in‑place* ABI expected by the harness: it mutates the ``bins``
array and returns ``None``.  All imports and the JIT warm‑up happen at module import
time, which is outside the timed region, so the compilation cost does not affect the
benchmark score.
"""

from __future__ import annotations

import numpy as np
import numba

__all__ = ["scatter_accum_dup"]

# ---------------------------------------------------------------------------
# Numba implementation – compiled once on import.
# ---------------------------------------------------------------------------
@numba.njit
def _scatter_numba(bins: np.ndarray, src: np.ndarray, ip: np.ndarray) -> None:
    """In‑place accumulation using a simple JIT‑compiled loop.

    This mirrors the reference ``for i in range(LEN_1D): bins[ip[i]] += src[i]`` but runs
    at compiled speed.  The loop is deliberately serial; Numba's parallel loops would
    introduce data races on duplicate indices.
    """
    for i in range(ip.shape[0]):
        bins[ip[i]] += src[i]

# Warm‑up the JIT compiler with a tiny dummy call so that the compilation cost is not
# measured during the timed benchmark runs.
_dummy_len = 1024
_dummy_bins = np.empty(_dummy_len, dtype=np.float64)
_dummy_src = np.empty(_dummy_len, dtype=np.float64)
_dummy_ip = np.arange(_dummy_len, dtype=np.int32)
_scatter_numba(_dummy_bins, _dummy_src, _dummy_ip)


def scatter_accum_dup(bins: np.ndarray, src: np.ndarray, ip: np.ndarray, LEN_1D: int) -> None:
    """Accumulate ``src`` into ``bins`` at positions given by ``ip``.

    Parameters
    ----------
    bins:
        1‑D output array to be updated in‑place.  Its length is ``LEN_1D``.
    src:
        1‑D input array of values to add, same length as ``ip``.
    ip:
        1‑D integer index array (``int32``) with values in ``[0, LEN_1D)``.
    LEN_1D:
        Logical length of the arrays – provided by the harness but not required for the
        implementation.  It is kept in the signature for compatibility.
    """
    # Delegate to the JIT‑compiled helper.  ``LEN_1D`` is ignored because the arrays
    # already carry their length information.
    _scatter_numba(bins, src, ip)
    # No explicit return; returning ``None`` signals the in‑place ABI.
    return None
