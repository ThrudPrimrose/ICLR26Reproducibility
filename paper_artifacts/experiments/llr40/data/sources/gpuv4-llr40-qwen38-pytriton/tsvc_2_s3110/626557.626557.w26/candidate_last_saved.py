"""Optimized TSVC tsvc_2 s3110 (first-occurrence global argmax of a 2-D array).

chksum = max(aa) + xindex + yindex where (xindex, yindex) is the FIRST
row-major occurrence of the maximum (reference uses strict `>`).

Strategy: read the array with one pinned worker thread per allowed CPU.
Each worker runs NumPy's (AVX-512 vectorized, GIL-released) argmax over its
slice; slice sizes are proportional to per-worker bandwidths measured once
at import time, so all workers finish together.  Results are combined in
ascending slice order with strict `>`, which preserves first-occurrence
semantics.  Everything (thread start, bandwidth probe, numpy code paths)
happens at import time; the timed call only submits and joins.
"""

import os
import queue
import threading
import time

import numpy as np

# ---------------------------------------------------------------------------
# worker pool: one thread pinned to one allowed CPU, persistent queues
# ---------------------------------------------------------------------------

def _allowed_cpus():
    try:
        return sorted(os.sched_getaffinity(0))
    except Exception:
        n = os.cpu_count() or 1
        return list(range(n))


class _PinnedPool(object):
    def __init__(self, cpus):
        self.n = len(cpus)
        self.cpus = cpus
        self.weight = [1.0] * self.n
        self._queues = [queue.Queue() for _ in cpus]
        self._threads = []
        for i in range(self.n):
            th = threading.Thread(target=self._run, args=(i,), daemon=True)
            th.start()
            self._threads.append(th)

    def _run(self, i):
        try:
            os.sched_setaffinity(0, {self.cpus[i]})
        except Exception:
            pass
        q = self._queues[i]
        while True:
            job, slot = q.get()
            try:
                slot[0] = job()
            except Exception as e:  # pragma: no cover
                slot[0] = e
            finally:
                slot[1].set()

    def map(self, jobs):
        """jobs: list of callables, one per worker (ascending segment order)."""
        slots = [(None, threading.Event()) for _ in jobs]
        for i, j in enumerate(jobs):
            self._queues[i].put((j, slots[i]))
        for _, ev in slots:
            ev.wait()
        out = [s[0] for s in slots]
        for e in out:
            if isinstance(e, BaseException):
                raise e
        return out


def _make_pool():
    cpus = _allowed_cpus()
    pool = _PinnedPool(cpus)
    # probe per-worker bandwidth on a 1 GB array (not timed)
    try:
        probe_n = 125_000_000  # 1 GB float64
        probe = np.random.rand(probe_n)
        sz = probe_n // pool.n
        segs = [(i * sz, min((i + 1) * sz, probe_n)) for i in range(pool.n)]

        def _time_seg(s, e):
            t0 = time.perf_counter()
            np.argmax(probe[s:e])
            return (e - s) * 8.0 / max(1e-9, time.perf_counter() - t0)

        for _ in range(2):  # first warms, second measures
            pool.weight = pool.map([lambda s, e: _time_seg(s, e) for s, e in segs])
    except Exception:
        pass
    # warm the exact call path once (small, contiguous + nontrivial)
    try:
        pool.map([lambda: np.argmax(probe[: 1 << 20]) for _ in range(pool.n)])
    except Exception:
        pass
    return pool


_POOL = _make_pool()
_NT = _POOL.n

# threshold below which threading overhead is not worth it (~single-thread
# numpy argmax runs at ~35 GB/s, thread fan-in costs ~0.3 ms)
_MT_ELEM = 16_000_000


def _seg_argmax(af, s, e):
    seg = af[s:e]
    k = int(np.argmax(seg))
    return seg[k], s + k


def s3110(aa, bb, LEN_2D):
    n = int(LEN_2D)
    m = int(aa.shape[0]) if aa.ndim > 1 else 1
    tot = m * n
    if aa.ndim == 1:
        af = aa.reshape(-1) if aa.flags['C_CONTIGUOUS'] else np.ascontiguousarray(aa).ravel()
    elif aa.flags['C_CONTIGUOUS']:
        af = aa.reshape(-1)
    else:
        af = np.ascontiguousarray(aa).ravel()

    if tot <= _MT_ELEM or _NT <= 1:
        k = int(np.argmax(af))
        maxv = float(af[k])
        bi = k
    else:
        w = _POOL.weight
        wsum = float(sum(w))
        b = [0] * (_NT + 1)
        for t in range(_NT - 1):
            b[t + 1] = min(tot, int(round(tot * w[t] / wsum)))
        b[_NT] = tot
        b[-1] = tot
        # ensure strictly monotone (tiny-weight edge)
        for t in range(_NT):
            if b[t + 1] <= b[t]:
                b[t + 1] = b[t] + 1
        for t in range(_NT - 1, 0, -1):
            if b[t + 1] < b[t]:
                b[t] = b[t + 1] - 1
        jobs = [(_seg_argmax, (af, b[t], b[t + 1])) for t in range(_NT)]
        res = _POOL.map([lambda args=jobs[t][1]: jobs[0][0](*args) for t in range(_NT)])
        maxv, bi = res[0]
        for t in range(1, _NT):
            v, k = res[t]
            if v > maxv:
                maxv, bi = v, k

    i, j = divmod(int(bi), n)
    bb[0, 0] = float(maxv) + float(i) + float(j)
    return None


tsvc_2_s3110 = s3110
tsvc_2_s3110_fp64 = s3110
