import numpy as np
import numba

@numba.njit(parallel=True, fastmath=True)
def s2710(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, e: np.ndarray, x: np.ndarray, LEN_1D: int) -> None:
    cond2 = LEN_1D > 10
    cond3 = x[0] > 0.0
    for i in numba.prange(LEN_1D):
        if a[i] > b[i]:
            a[i] += b[i] * d[i]
            if cond2:
                c[i] += d[i] * d[i]
            else:
                c[i] = d[i] * e[i] + 1.0
        else:
            b[i] = a[i] + e[i] * e[i]
            if cond3:
                c[i] = a[i] + d[i] * d[i]
            else:
                c[i] += e[i] * e[i]
