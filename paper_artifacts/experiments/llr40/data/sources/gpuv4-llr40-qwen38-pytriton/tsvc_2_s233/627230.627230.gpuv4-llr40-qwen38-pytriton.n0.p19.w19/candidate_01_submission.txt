import numpy as np
import numba as nb

BLK = 16


@nb.njit(parallel=True)
def _scan_cols(aa, cc, N, BLK):
    nblk = (N - 8 + BLK - 1) // BLK
    for b in nb.prange(nblk):
        i0 = 8 + b * BLK
        m = min(BLK, N - i0)
        run = np.empty(BLK, dtype=np.float64)
        for k in range(m):
            run[k] = aa[7, i0 + k]
        for j in range(8, N):
            for k in range(m):
                run[k] += cc[j, i0 + k]
                aa[j, i0 + k] = run[k]


@nb.njit(parallel=True)
def _scan_rows(bb, cc, N):
    for j in nb.prange(8, N):
        v = bb[j, 7]
        for i in range(8, N):
            v += cc[j, i]
            bb[j, i] = v


def _warm():
    d = np.zeros((24, 24))
    _scan_cols(d, d, 24, 16)
    _scan_rows(d, d, 24)


_warm()


def _ok(a):
    return a.dtype == np.float64 and a.flags["C_CONTIGUOUS"]


def s233(aa, bb, cc, LEN_2D):
    n = int(LEN_2D)
    if n <= 8:
        return None
    if not (_ok(aa) and _ok(bb) and _ok(cc)):
        aa = np.ascontiguousarray(aa, dtype=np.float64)
        bb = np.ascontiguousarray(bb, dtype=np.float64)
        cc = np.ascontiguousarray(cc, dtype=np.float64)
    _scan_cols(aa, cc, n, BLK)
    _scan_rows(bb, cc, n)
    return None
