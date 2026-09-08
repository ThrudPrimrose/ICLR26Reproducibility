# TSVC tsvc_2 s2233 -- optimized python implementation.
#
# Math: for i,j in 8..N-1:
#   aa[j, i] = aa[j-1, i] + cc[j, i]   -> per-column (i) inclusive scan over rows
#   bb[i, j] = bb[i-1, j] + cc[i, j]   -> per-column (j) inclusive scan over rows
# Final values, for column k >= 8 and row r >= 8:
#   aa[r, k] = aa[7, k] + sum_{t=8..r} cc[t, k]
#   bb[r, k] = bb[7, k] + sum_{t=8..r} cc[t, k]
# i.e. BOTH outputs are the column-wise row-scan of the 8x8 subblock of cc,
# with different seed rows (row 7 of aa / bb).  Rows < 8 and cols < 8 untouched.

import numpy as np

_IMPORT_ERRS = []

# ------------------------- numpy path (small sizes) -------------------------

def _numpy_path(aa, bb, cc, N):
    s = slice(8, N)
    tmp = np.cumsum(cc[s, s], axis=0)
    np.add(tmp, aa[7:8, s], out=aa[s, s])
    np.add(aa[s, s], bb[7:8, s] - aa[7:8, s], out=bb[s, s])

# ------------------------- GPU (triton) path --------------------------------
# NOTE: torch/HIP must be initialized BEFORE numba's parallel workqueue is
# started, otherwise later GPU operations segfault on this platform.

_GPU_OK = False
_ROWS = 64
_COLS = 16
_NW = 2

try:
    import torch
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["N", "C"])
    def _scan2d(cc_p, aa_p, bb_p, N, C, ROWS: tl.constexpr, COLS: tl.constexpr):
        g = tl.program_id(0)
        c0 = g * COLS
        offs_c = tl.arange(0, COLS)
        cc = c0 + offs_c
        mc = cc < C
        ra = tl.load(aa_p + 7 * N + 8 + cc, mask=mc, other=0.0)
        rb = tl.load(bb_p + 7 * N + 8 + cc, mask=mc, other=0.0)
        offs_r = tl.arange(0, ROWS)
        for r0 in range(0, C, ROWS):
            rr = r0 + offs_r
            mr = rr < C
            m = mr[:, None] & mc[None, :]
            idx = (rr[:, None] + 8) * N + (cc[None, :] + 8)
            v = tl.load(cc_p + idx, mask=m, other=0.0)
            cs = tl.cumsum(v, axis=0)
            tl.store(aa_p + idx, cs + ra[None, :], mask=m)
            tl.store(bb_p + idx, cs + rb[None, :], mask=m)
            tot = tl.sum(v, axis=0)
            ra += tot
            rb += tot

    def _gpu_path(aa, bb, cc, N, C):
        cc_d = torch.from_numpy(np.ascontiguousarray(cc)).cuda()
        aa_d = torch.empty_like(cc_d)
        bb_d = torch.empty_like(cc_d)
        aa_d[7].copy_(torch.from_numpy(np.ascontiguousarray(aa[7])).cuda())
        bb_d[7].copy_(torch.from_numpy(np.ascontiguousarray(bb[7])).cuda())
        aa_d[:, :8].copy_(torch.from_numpy(aa[:, :8]).cuda())
        bb_d[:, :8].copy_(torch.from_numpy(bb[:, :8]).cuda())
        _scan2d[(triton.cdiv(C, _COLS),)](
            cc_d, aa_d, bb_d, N, C, ROWS=_ROWS, COLS=_COLS, num_warps=_NW)
        torch.from_numpy(aa)[8:, :].copy_(aa_d[8:, :])
        torch.from_numpy(bb)[8:, :].copy_(bb_d[8:, :])

    if torch.cuda.is_available():
        _d = torch.zeros(64, 64, device="cuda", dtype=torch.float64)
        _scan2d[(triton.cdiv(56, _COLS),)](_d, _d, _d, 64, 56,
                                           ROWS=_ROWS, COLS=_COLS, num_warps=_NW)
        torch.cuda.synchronize()
        _h = np.zeros(64 * 64, dtype=np.float64)
        _h2 = torch.from_numpy(_h).cuda()
        torch.cuda.synchronize()
        _GPU_OK = True
except Exception:
    import traceback
    _IMPORT_ERRS.append("gpu import: " + traceback.format_exc())
    _GPU_OK = False

# ------------------------- numba path (CPU fallback) ------------------------

_NumbaK = None
try:
    from numba import njit, prange

    @njit(parallel=True, cache=False)
    def _k_cols(aa, bb, cc, N):
        C = N - 8
        for col in prange(C):
            j = 8 + col
            s_a = aa[7 * N + j]
            s_b = bb[7 * N + j]
            for i in range(8, N):
                p = i * N + j
                v = cc[p]
                s_a += v
                aa[p] = s_a
                s_b += v
                bb[p] = s_b

    _NumbaK = _k_cols
    try:
        # compile at import time (untimed); tiny call
        _a = np.zeros((64, 64)); _b = np.zeros((64, 64)); _c = np.zeros((64, 64))
        _k_cols(_a, _b, _c, 64)
    except Exception:
        _IMPORT_ERRS.append("numba warm: " + traceback.format_exc())
except Exception:
    import traceback
    _IMPORT_ERRS.append("numba import: " + traceback.format_exc())
    _NumbaK = None

# ------------------------- dispatch -----------------------------------------

def s2233(aa, bb, cc, LEN_2D):
    N = int(LEN_2D)
    if N <= 8:
        return None
    C = N - 8
    if C < 1536:
        _numpy_path(aa, bb, cc, N)
    elif _GPU_OK:
        try:
            _gpu_path(aa, bb, cc, N, C)
        except Exception:
            _IMPORT_ERRS.append("gpu call: " + traceback.format_exc())
            if _NumbaK is not None:
                _numba_path(aa, bb, cc, N)
            else:
                _numpy_path(aa, bb, cc, N)
    elif _NumbaK is not None:
        _numba_path(aa, bb, cc, N)
    else:
        _numpy_path(aa, bb, cc, N)
    return None

def _numba_path(aa, bb, cc, N):
    _NumbaK(aa, bb, cc, N)
