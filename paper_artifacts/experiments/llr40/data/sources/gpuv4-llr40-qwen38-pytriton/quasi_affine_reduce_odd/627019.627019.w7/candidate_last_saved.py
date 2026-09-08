"""Optimized TSVC tsvc_2_5 ``quasi_affine_reduce_odd``: out[0] = sum(a[i] for i in 1,3,...).

Strategy (decided once per process at the first large call, which the harness runs as a
discarded warmup rep):
  * tiny n          -- single-threaded fastmath numba (matches the numba baseline, no launch cost)
  * medium n        -- numba prange across the slot's physical cores
  * large n         -- the faster of {prange, GPU H2D + device reduce}, picked by probing both
                       once; the GPU path is frequency-independent (DMA engine), prange scales
                       with the CPU frequency state, so an adaptive pick is robust.
All first-call costs (numba JIT, torch/HIP context, allocator pools) happen at import time,
before the harness clock starts.
"""
import os
import time

import numpy as np
import numba
from numba import njit, prange

try:
    _NPROC = len(os.sched_getaffinity(0))
except Exception:  # noqa: BLE001
    _NPROC = os.cpu_count() or 1
try:
    numba.set_num_threads(max(1, min(64, _NPROC)))
except Exception:  # noqa: BLE001
    pass


@njit(fastmath=True)
def _seq(a, out, n):
    acc = 0.0
    for i in range(1, n, 2):
        acc += a[i]
    out[0] = acc


@njit(parallel=True, fastmath=True)
def _par(a, out, n):
    m = (n + 1) // 2
    acc = 0.0
    for k in prange(m):
        acc += a[1 + 2 * k]
    out[0] = acc


# ---- compile both CPU paths at import (untimed) ----
_tmp_out = np.zeros(1)
_dummy = np.zeros(128)
_seq(_dummy, _tmp_out, 128)
_par(_dummy, _tmp_out, 128)

# ---- GPU path (optional; disabled transparently if unavailable) ----
import torch  # noqa: E402

_DEV = None
try:
    if torch.cuda.is_available():
        _DEV = torch.device("cuda:0")
        _ph = torch.empty(1 << 20, dtype=torch.float64)
        _pd = _ph.to(_DEV)
        _pd[1::2].sum().item()
        del _ph, _pd
        torch.cuda.synchronize()
except Exception:  # noqa: BLE001
    _DEV = None


def _gpu(a, out, n):
    ta = torch.from_numpy(a)
    x = ta[:n].to(_DEV, non_blocking=False)
    out[0] = x[1::2].sum().item()


_ST = {"probed_n": None, "use_gpu": False, "gpu_ok": _DEV is not None}
_N_SEQ = 1 << 17      # below this: single thread
_N_PROBE = 1 << 24    # at/above this: probe CPU vs GPU once, then stick with the winner


def quasi_affine_reduce_odd(a, out, LEN_1D):
    n = a.shape[0]
    if LEN_1D is not None and LEN_1D < n:
        n = int(LEN_1D)
    if n < 2:
        out[0] = 0.0
        return None
    if n <= _N_SEQ:
        _seq(a, out, n)
        return None
    if _DEV is not None and n >= _N_PROBE:
        st = _ST
        if st["probed_n"] != n and st["gpu_ok"]:
            best_cpu = 1e18
            for _ in range(3):
                t0 = time.perf_counter()
                _par(a, out, n)
                best_cpu = min(best_cpu, time.perf_counter() - t0)
            best_gpu = 1e18
            try:
                for _ in range(3):
                    t0 = time.perf_counter()
                    _gpu(a, out, n)
                    best_gpu = min(best_gpu, time.perf_counter() - t0)
            except Exception:  # noqa: BLE001
                best_gpu = 1e18
            st["use_gpu"] = best_gpu < best_cpu * 1.05
            st["probed_n"] = n
        if st["use_gpu"]:
            try:
                _gpu(a, out, n)
                return None
            except Exception:  # noqa: BLE001
                st["gpu_ok"] = False
    _par(a, out, n)
    return None
