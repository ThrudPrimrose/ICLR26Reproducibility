import os
import numba as nb


def _pick_threads():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return os.cpu_count() or 1


@nb.njit(parallel=True, cache=True)
def _s4112_pt(a, b, ip, N):
    for i in nb.prange(N):
        a[i] = a[i] + b[ip[i]] * 2.0


@nb.njit(nogil=True, cache=True)
def _s4112_st(a, b, ip, N):
    for i in range(N):
        a[i] = a[i] + b[ip[i]] * 2.0


def s4112(a, b, ip, LEN_1D):
    n = int(LEN_1D)
    if n <= 4096:
        _s4112_st(a, b, ip, n)
    else:
        try:
            nb.set_num_threads(_pick_threads())
        except Exception:
            pass
        _s4112_pt(a, b, ip, n)
    return None
