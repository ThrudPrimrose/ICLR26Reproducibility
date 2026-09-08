'''Optimized implementation of the TSVC ``s152`` kernel.

The reference implementation (see ``/shared/tasks/tsvc_2_s152/tsvc_2_s152_numpy.py``) uses
explicit Python ``for`` loops to compute::

    for i in range(LEN_1D):
        b[i] = d[i] * e[i]
    for i in range(LEN_1D):
        a[i] = a[i] + b[i] * c[i]

While functionally correct, these loops incur Python overhead for each iteration.  On the
Python track the baseline is the same algorithm JIT‑compiled with *Numba*, so to obtain a
speed‑up we must both reduce the overhead and give the JIT more chances for vectorisation.

The strategy is simple:

* Fuse the two passes into a single loop – each iteration computes ``tmp = d[i] * e[i]``
  and stores it in ``b[i]`` while also updating ``a[i]`` with ``tmp * c[i]``.
* Use ``numba.njit`` with ``parallel=True`` and ``fastmath=True`` so the loop can be
  automatically parallelised and aggressively vectorised.
* Perform a trivial warm‑up call at import time.  Import time is *outside* the timed region
  of the benchmark, so the compilation cost of the JIT does not affect the measured
  runtime.

The function signature matches the reference exactly – the kernel is called as
``s152(a, b, c, d, e, LEN_1D)`` where all array arguments are ``numpy.ndarray`` objects of
type ``float64`` (the reference uses double precision).  The kernel mutates ``a`` and ``b``
in‑place and returns ``None`` (the default Python return value).
'''

from __future__ import annotations

import numpy as np
import numba


@numba.njit(parallel=True, fastmath=True)
def s152(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, e: np.ndarray, LEN_1D: int) -> None:
    '''Fused, parallel version of the ``s152`` kernel.

    Parameters
    ----------
    a, b, c, d, e : 1-dimensional ``float64`` NumPy arrays.
        ``a`` and ``b`` are updated in place.
    LEN_1D : int
        Logical length of the vectors; only the first LEN_1D elements are processed.
    '''
    for i in numba.prange(LEN_1D):
        tmp = d[i] * e[i]
        b[i] = tmp
        a[i] = a[i] + tmp * c[i]

# Warm‑up compilation to ensure JIT overhead is not measured.
_dummy = np.empty(0, dtype=np.float64)
s152(_dummy, _dummy, _dummy, _dummy, _dummy, 0)
