# TSVC s332 ext_break_capture: find first i with a[i] > K, capture i and a[i].
#
# Strategy: the answer lives at the first element > K; to be sure, every element before it must
# be inspected, so the irreducible cost is streaming ~8 * cut bytes. We stream at aggregate
# memory bandwidth: a thread pool (one worker per physical core) computes per-chunk maxima
# (numpy releases the GIL inside its C kernels), results are consumed in index order and the
# scan stops at the first chunk whose max exceeds K; one final compare+argmax inside that
# chunk pinpoints the index. Chunk size trades dispatch overhead against in-flight overshoot.
import os

import numpy as np


def _physical_cores():
    """One worker per physical core (drop SMT siblings) within the process affinity."""
    try:
        aff = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        return os.cpu_count() or 1
    chosen, seen = [], set()
    for cpu in sorted(aff):
        try:
            with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as f:
                core = min(int(x) for x in f.read().split(",") if x)
        except (OSError, ValueError):
            core = cpu
        if core not in seen:
            seen.add(core)
            chosen.append(cpu)
    return len(chosen) or len(aff)


_NWORKERS = max(1, _physical_cores())

from concurrent.futures import ThreadPoolExecutor  # noqa: E402

_POOL = ThreadPoolExecutor(max_workers=_NWORKERS)

_CHUNKS_PER_WORKER = 16  # overshoot vs dispatch-overhead balance at the graded size


def ext_break_capture(a, out_index, out_value, LEN_1D, K):
    n = int(LEN_1D)
    if n <= 0:
        out_index[0] = -1
        out_value[0] = -1.0
        return None
    K = float(K)
    if not (a.flags["C_CONTIGUOUS"] and a.dtype == np.float64):
        a = np.ascontiguousarray(a, dtype=np.float64)

    if n < (1 << 16):
        m = a[:n] > K
        i = int(m.argmax())
        if m[i]:
            out_index[0] = i
            out_value[0] = a[i]
        else:
            out_index[0] = -1
            out_value[0] = -1.0
        return None

    T = _NWORKERS
    C = max(1 << 20, -(-n // (_CHUNKS_PER_WORKER * T)))
    first_lo = -1
    for lo, mx in zip(range(0, n, C), _POOL.map(lambda s: a[s : s + C].max(), range(0, n, C))):
        if mx > K:
            first_lo = lo
            break
    if first_lo < 0:
        out_index[0] = -1
        out_value[0] = -1.0
        return None
    hi = min(first_lo + C, n)
    m = a[first_lo:hi] > K
    i = first_lo + int(m.argmax())
    out_index[0] = i
    out_value[0] = a[i]
    return None
