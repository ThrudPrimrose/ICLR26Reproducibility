"""TSVC tsvc_2 s316 -- min reduction. Python arm (numba prange, warmed at import)."""
import os
import time
import numpy as np
import numba
from numba import njit, prange


@njit(parallel=True)
def _pmin(a, part, t):
    n = a.shape[0]
    h = (n + t - 1) // t
    for j in prange(t):
        lo = j * h
        hi = lo + h
        if hi > n:
            hi = n
        if hi <= lo:
            part[j] = np.inf
        else:
            x = a[lo]
            for i in range(lo, hi):
                x = min(x, a[i])
            part[j] = x


@njit()
def _seq(a):
    x = a[0]
    n = a.shape[0]
    for i in range(1, n):
        x = min(x, a[i])
    return x


_part = np.zeros(1024)


def _choose_threads(scratch, cand_t, iters=2):
    best = (None, float("inf"))
    rows = []
    errs = []
    for t in cand_t:
        try:
            numba.set_num_threads(t)
            _pmin(scratch, _part, t)  # warm
            b = float("inf")
            for _ in range(iters):
                t0 = time.perf_counter()
                _pmin(scratch, _part, t)
                b = min(b, time.perf_counter() - t0)
        except Exception as e:
            errs.append((t, repr(e)))
            continue
        rows.append((t, b))
        if b < best[1]:
            best = (t, b)
    if best[0] is None:
        raise RuntimeError("no thread count worked: %r" % (errs,))
    return best, rows, errs


_scratch = np.random.default_rng(0).standard_normal(1 << 27)  # 1.07 GB
_rows = []
_errs = []
try:
    import os as _os
    _cc = _os.cpu_count() or 1
    _best_t, _rows, _errs = _choose_threads(_scratch, sorted({min(t, _cc) for t in (16, 24, 32, 48, 96, 192)}))
    numba.set_num_threads(_best_t)
    _T = _best_t
except Exception:
    import traceback; print("IMPORT-BENCH-FAIL\n" + traceback.format_exc(), flush=True)
    _T = numba.config.NUMBA_NUM_THREADS
_seq(np.ones(1 << 16))


def s316(a, result, LEN_1D):
    n = a.shape[0]
    if n < 1 << 16:
        x = _seq(a)
    else:
        _pmin(a, _part, _T)
        x = _part[0]
        for j in range(1, _T):
            x = min(x, _part[j])
    result[0] = x
    return None


# ---- probe block (removed later) ----
def _probe(a, result, x, t_par):
    try:
        n = a.shape[0]
        print("PROBE env_NUMBA_NUM_THREADS=%s chosen_T=%d" %
              (os.environ.get("NUMBA_NUM_THREADS"), _T), flush=True)
        print("PROBE t_bench: " + (" ".join("t=%d:%.1fms" % (t, b * 1e3) for t, b in _rows) or "EMPTY"),
              flush=True)
        print("PROBE errs: %r" % (_errs,), flush=True)
        print("PROBE n=%d t_par_full=%.2fms effBW=%.1f GB/s result=%r" %
              (n, t_par * 1e3, n * 8 / t_par / 1e9, x), flush=True)
    except Exception as e:
        print("PROBE-ERR %r" % e, flush=True)


def s316_probe(a, result, LEN_1D):
    n = a.shape[0]
    if n < 1 << 16:
        x = _seq(a)
        t_par = 0.0
    else:
        t0 = time.perf_counter()
        _pmin(a, _part, _T)
        x = _part[0]
        for j in range(1, _T):
            x = min(x, _part[j])
        t_par = time.perf_counter() - t0
    result[0] = x
    _probe(a, result, x, t_par)
    return None


s316 = s316_probe
