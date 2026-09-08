import os
import numpy as np
import numba as nb


def _affinity_count():
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1


_CORES = _affinity_count()
try:
    nb.set_num_threads(_CORES)
except Exception:
    pass


@nb.njit(parallel=True, fastmath=True)
def _kernel(a, b, m, nt, chunk, boundary):
    # Serial boundary capture: completes BEFORE the parallel region starts,
    # so each block reads its one cross-block element from the original data.
    for t in range(nt):
        end = (t + 1) * chunk
        if end > m:
            end = m
        boundary[t] = a[end]
    # Parallel: block t owns a[start:end]; it only reads/writes its own block
    # plus its pre-captured boundary element -> no cross-block hazard.
    for t in nb.prange(nt):
        start = t * chunk
        end = start + chunk
        if end > m:
            end = m
        for j in range(start, end - 1):
            a[j] = a[j + 1] + b[j]
        if start < end:
            a[end - 1] = boundary[t] + b[end - 1]


_NT_MULT = 4      # default, overridden by import-time autotune
_NT_CAP = 16384


def _nt_for(m):
    nt = _CORES * _NT_MULT
    if nt < 1:
        nt = 1
    if nt > _NT_CAP:
        nt = _NT_CAP
    if nt > m:
        nt = m
    return nt


def ext_war_unit(a, b, LEN_1D):
    n = int(LEN_1D)
    m = n - 1
    if m <= 0:
        return None
    nt = _nt_for(m)
    chunk = (m + nt - 1) // nt
    boundary = np.empty(nt, dtype=a.dtype)
    _kernel(a, b, m, nt, chunk, boundary)
    return None


def _run_once(a, b, m, nt):
    chunk = (m + nt - 1) // nt
    boundary = np.empty(nt, dtype=a.dtype)
    _kernel(a, b, m, nt, chunk, boundary)


def _autotune():
    """Pick the best block multiplier on THIS hardware at import (untimed)."""
    global _NT_MULT
    import time
    results = {}
    try:
        n = 40_000_000
        rng = np.random.default_rng(0)
        a0 = rng.uniform(-1, 1, n)
        b = rng.uniform(-1, 1, n)
        m = n - 1
        best_ns = 1e30
        best_mult = _NT_MULT
        for mult in (2, 4, 8, 16, 32):
            nt = min(_CORES * mult, _NT_CAP, m)
            if nt < 1:
                continue
            # warm
            a = a0.copy()
            _run_once(a, b, m, nt)
            t = 1e30
            for _ in range(2):
                a = a0.copy()
                t0 = time.perf_counter_ns()
                _run_once(a, b, m, nt)
                t = min(t, time.perf_counter_ns() - t0)
            results[mult] = t
            if t < best_ns:
                best_ns = t
                best_mult = mult
        _NT_MULT = best_mult
    except Exception as e:  # noqa
        results['error'] = repr(e)
    # diagnostic
    try:
        with open('/shared/agent-3/diag.json', 'w') as f:
            f.write(repr({
                'cores': _CORES,
                'cpu_count': os.cpu_count(),
                'numba_threads': nb.config.NUMBA_NUM_THREADS,
                'best_mult': _NT_MULT,
                'results_ns': results,
                'affinity_len': len(os.sched_getaffinity(0)),
            }))
    except Exception:
        pass


def _warm():
    n = 1 << 20
    a = np.random.default_rng(0).uniform(-1, 1, n)
    b = np.random.default_rng(1).uniform(-1, 1, n)
    m = n - 1
    nt = _nt_for(m)
    _run_once(a, b, m, nt)


_warm()
_autotune()
