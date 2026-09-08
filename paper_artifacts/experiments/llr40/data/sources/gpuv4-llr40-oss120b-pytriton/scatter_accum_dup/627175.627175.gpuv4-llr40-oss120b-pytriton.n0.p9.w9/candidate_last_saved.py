'''Hybrid NumPy implementation for ``scatter_accum_dup``.

The kernel must correctly accumulate ``src`` values into ``bins`` at the
indices specified by ``ip``.  ``ip`` may contain duplicates, so a simple
vectorised ``bins[ip] += src`` would lose updates.  NumPy's ``np.add.at``
provides the required semantics and is fast for modest problem sizes.

For very large inputs the overhead of NumPy's ufunc can become noticeable.
In that regime we fall back to a Numba-compiled loop, which operates directly on
the raw memory and typically runs faster.  The Numba path is imported lazily
to avoid the import cost when it is not needed.
''' 
import numpy as np

# Threshold (elements) where the Numba implementation becomes beneficial.
# Empirically chosen; adjust if the benchmark's input sizes differ.
_THRESHOLD = 30_000_000

# Placeholder for the compiled Numba function – initialised on first use.
_scatter_numba = None

def _ensure_numba():
    """Import Numba and compile the fallback implementation if not already.
    The function is called only for problem sizes exceeding ``_THRESHOLD``.
    """
    global _scatter_numba
    if _scatter_numba is None:
        import numba
        @numba.njit(fastmath=True, cache=True)
        def _inner(bins: np.ndarray, src: np.ndarray, ip: np.ndarray, LEN: int) -> None:
            for i in range(LEN):
                bins[ip[i]] = bins[ip[i]] + src[i]
        # Warm‑up compilation with a trivial zero‑length call.
        _inner(np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int32), 0)
        _scatter_numba = _inner
    return _scatter_numba

def scatter_accum_dup(bins: np.ndarray, src: np.ndarray, ip: np.ndarray, LEN_1D: int) -> None:
    """Accumulate ``src`` into ``bins`` using ``ip`` as indices.

    Parameters
    ----------
    bins : np.ndarray
        Output buffer (modified in‑place).
    src : np.ndarray
        Values to add.
    ip : np.ndarray
        Integer indices into ``bins``; may contain duplicates.
    LEN_1D : int
        Declared length of the arrays.
    """
    if LEN_1D == 0:
        return
    if LEN_1D > _THRESHOLD:
        # Use the Numba implementation for large workloads.
        _ensure_numba()(bins, src, ip, LEN_1D)
    else:
        # NumPy's ``add.at`` handles duplicates correctly and is fast for
        # moderate sizes.
        np.add.at(bins, ip, src)
