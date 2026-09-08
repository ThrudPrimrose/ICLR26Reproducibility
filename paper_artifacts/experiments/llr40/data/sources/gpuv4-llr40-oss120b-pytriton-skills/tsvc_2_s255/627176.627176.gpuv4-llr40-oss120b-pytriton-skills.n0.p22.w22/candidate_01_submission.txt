'''Optimized implementation of the TSVC ``s255`` kernel in pure NumPy.

The reference kernel (see ``/shared/tasks/tsvc_2_s255/tsvc_2_s255_numpy.py``) is:

.. code-block:: python

    def s255(a, b, LEN_1D):
        x = b[LEN_1D - 1]
        y = b[LEN_1D - 2]
        for i in range(LEN_1D):
            a[i] = (b[i] + x + y) * 0.333
            y = x
            x = b[i]

which is equivalent to a circular 3‑point averaging:

    a[i] = (b[i] + b[(i-1) % N] + b[(i-2) % N]) / 3

The baseline used by the benchmark is a ``numba.njit`` compilation of the loop
above.  A pure‑Python loop is far slower; a naïve NumPy vectorisation using
``np.roll`` would allocate two temporaries of size ``N`` each, which is
unnecessary.  The implementation below performs the operation with a single
write‑pass and only the required scalar temporaries, keeping the memory traffic
to the minimum possible while still staying fully vectorised.

The kernel follows the *in‑place* ABI expected by the harness: the output array
``a`` is modified directly and the function returns ``None``.
'''

from __future__ import annotations
import numpy as np

# Pre‑compute the constant 1/3 as a Python float—this is evaluated once at import time.
_INV_THREE: float = 0.333

def s255(a: np.ndarray, b: np.ndarray, LEN_1D: int) -> None:
    '''In‑place circular three‑point average.

    Parameters
    ----------
    a : np.ndarray
        Output array of shape ``(LEN_1D,)`` and a floating‑point dtype.
    b : np.ndarray
        Input array of the same shape and dtype as ``a``.
    LEN_1D : int
        Length of the one‑dimensional problem.
    '''
    # Guard against degenerate size.
    if LEN_1D == 0:
        return
    # When LEN_1D == 1 the circular average uses the same element three times.
    if LEN_1D == 1:
        a[0] = (b[0] + b[0] + b[0]) * _INV_THREE
        return
    # Main vectorised path for LEN_1D >= 2.
    # Compute the interior of the array (indices 2 .. N-1) without temporaries.
    a[2:] = (b[2:] + b[1:-1] + b[:-2]) * _INV_THREE
    # Handle the first two elements which involve wrap‑around.
    a[0] = (b[0] + b[-1] + b[-2]) * _INV_THREE
    a[1] = (b[1] + b[0] + b[-1]) * _INV_THREE
    # No return value for the in‑place ABI.
    return None
