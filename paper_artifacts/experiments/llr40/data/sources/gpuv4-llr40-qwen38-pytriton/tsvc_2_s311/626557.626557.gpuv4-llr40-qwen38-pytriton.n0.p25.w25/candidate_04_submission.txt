"""TSVC tsvc_2 s311: sum_out[0] = sum(a[0:LEN_1D]).

Torch CPU reduction: OpenMP pool (OMP_NUM_THREADS from the SLURM
allocation, OMP_PROC_BIND=close in this environment) does a vectorised
blocked reduction. OMP worker pool is created at import time so the
first timed call pays no pool-creation cost.
Falls back to a threaded numpy pool if torch is unavailable.
"""
import os
import numpy as np

_NT = int(os.environ.get("OMP_NUM_THREADS", "0") or 0)
if _NT <= 0:
    try:
        _NT = len(os.sched_getaffinity(0))
    except AttributeError:
        _NT = os.cpu_count() or 4
_NT = max(1, min(int(_NT), 64))

_d = np.zeros(8, dtype=np.float64)

_HAVE_TORCH = False
try:
    import torch
    torch.set_num_threads(_NT)
    _HAVE_TORCH = True
    # Warm: create the OpenMP worker pool before the clock starts.
    torch.from_numpy(_d).sum().item()
except Exception:
    _HAVE_TORCH = False

if not _HAVE_TORCH:
    from concurrent.futures import ThreadPoolExecutor

    _pool = ThreadPoolExecutor(max_workers=_NT)

    def _sum_sliced(arr, n, nt):
        chunk = (n + nt - 1) // nt
        futs = [_pool.submit(arr[i * chunk:min((i + 1) * chunk, n)].sum)
                for i in range(nt)]
        total = 0.0
        for f in futs:
            total += f.result()
        return total

    for _i in range(_NT):
        _pool.submit(_d.sum).result()


def s311(a, sum_out, LEN_1D):
    n = LEN_1D
    if n <= 0:
        sum_out[0] = 0.0
        return None
    if _HAVE_TORCH:
        t = torch.from_numpy(np.ascontiguousarray(a[:n]))
        sum_out[0] = t.sum().item()
    else:
        arr = np.ascontiguousarray(a[:n])
        nt = min(_NT, max(1, n // 5_000_000))
        if nt == 1:
            sum_out[0] = arr.sum()
        else:
            sum_out[0] = _sum_sliced(arr, n, nt)
    return None
