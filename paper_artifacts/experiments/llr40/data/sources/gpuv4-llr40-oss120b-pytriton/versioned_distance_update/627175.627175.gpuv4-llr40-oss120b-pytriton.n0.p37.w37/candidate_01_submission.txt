import numpy as np
import numba as nb
from typing import Any, Optional, Tuple

def initialize(LEN_1D: int, datatype: type = np.float64, variant_spec: Optional[Any] = None, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    a = rng.uniform(0.5, 2.0, LEN_1D).astype(datatype)
    b = rng.uniform(0.5, 1.5, LEN_1D).astype(datatype)
    c = rng.uniform(0.5, 1.5, LEN_1D).astype(datatype)
    return a, b, c

@nb.njit(parallel=True, fastmath=True)
def _versioned_distance_update_impl(a: np.ndarray, b: np.ndarray, c: np.ndarray, LEN_1D: int, K: int):
    for r in nb.prange(K):
        start = r + K
        for i in range(start, LEN_1D, K):
            a[i] = 0.75 * a[i - K] + b[i] * c[i]

def versioned_distance_update(a: np.ndarray, b: np.ndarray, c: np.ndarray, LEN_1D: int, K: int) -> None:
    _versioned_distance_update_impl(a, b, c, LEN_1D, K)
