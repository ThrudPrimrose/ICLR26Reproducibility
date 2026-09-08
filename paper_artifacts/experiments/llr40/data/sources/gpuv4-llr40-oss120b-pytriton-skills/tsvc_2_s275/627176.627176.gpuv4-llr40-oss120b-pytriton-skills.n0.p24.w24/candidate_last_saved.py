"""Hybrid Numba implementation of TSVC kernel s275.

The kernel updates ``aa`` column‑wise according to the recurrence

    aa[j, i] = aa[j-1, i] + bb[j, i] * cc[j, i]

for each column ``i`` where the first element ``aa[0, i]`` is positive.

* For modest problem sizes we stay on the CPU and use a simple Numba‑compiled
  loop, which incurs virtually no overhead beyond the Python call.
* For larger matrices the work becomes memory‑bandwidth bound; we exploit the
  independence of columns by parallelising the outer loop with Numba's
  ``prange``.

The function follows the in‑place ABI expected by the benchmark harness: it
mutates the supplied ``aa`` array and returns ``None``.
"""

import numpy as np
import numba

# ---------------------------------------------------------------------------
# Sequential implementation (used for small matrices where the parallel launch
# overhead would dominate).
# ---------------------------------------------------------------------------
@numba.njit(fastmath=True)
def _s275_seq(aa, bb, cc, N):
    mask = aa[0, :] > 0.0
    for i in range(N):
        if mask[i]:
            for j in range(1, N):
                aa[j, i] = aa[j - 1, i] + bb[j, i] * cc[j, i]

# ---------------------------------------------------------------------------
# Parallel implementation (leverages multiple CPU cores).
# ---------------------------------------------------------------------------
@numba.njit(parallel=True, fastmath=True)
def _s275_par(aa, bb, cc, N):
    mask = aa[0, :] > 0.0
    for i in numba.prange(N):
        if mask[i]:
            for j in range(1, N):
                aa[j, i] = aa[j - 1, i] + bb[j, i] * cc[j, i]

# ---------------------------------------------------------------------------
# Dispatcher selects the appropriate implementation based on the matrix size.
# ---------------------------------------------------------------------------
def s275(aa, bb, cc, LEN_2D):
    # Heuristic threshold: below this size the parallel launch overhead is
    # likely to outweigh any benefit.
    if LEN_2D <= 512:
        _s275_seq(aa, bb, cc, LEN_2D)
    else:
        _s275_par(aa, bb, cc, LEN_2D)
    return None
