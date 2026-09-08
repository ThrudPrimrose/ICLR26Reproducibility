# versioned_distance_update -- runtime-distance recurrence:
#   a[i] = 0.75 * a[i-K] + b[i] * c[i]   for i in K..LEN_1D-1  (in-place on a)
#
# The dependence distance K is a runtime value: the K residue classes mod K are
# independent chains, each a serial first-order recurrence.  We keep each chain's
# carry in a register (no load of a[i-K]) and run the chains with numba prange,
# so every K the manifest declares is handled by one kernel:
#   K=1    -> one serial chain (register carry, ~fma-latency bound)
#   K=5    -> 5 chains
#   K=251  -> 251 chains
#   K=4096 -> 4096 chains (block-parallel)
import numpy as np
from numba import njit, prange


@njit(parallel=True)
def _vdpu(a, b, c, LEN, K):
    for r in prange(K):
        x = a[r]
        for i in range(r + K, LEN, K):
            x = 0.75 * x + b[i] * c[i]
            a[i] = x


def versioned_distance_update(a, b, c, LEN_1D, K):
    _vdpu(a, b, c, int(LEN_1D), int(K))
    return None


# ---- import-time warmup: compile the numba kernel BEFORE the clock starts ----
_n = 16384
_a = np.random.uniform(0.5, 2.0, _n)
_b = np.random.uniform(0.5, 1.5, _n)
_c = np.random.uniform(0.5, 1.5, _n)
for _k in (1, 5, 251, 4096):
    _vdpu(_a, _b, _c, _n, _k)
