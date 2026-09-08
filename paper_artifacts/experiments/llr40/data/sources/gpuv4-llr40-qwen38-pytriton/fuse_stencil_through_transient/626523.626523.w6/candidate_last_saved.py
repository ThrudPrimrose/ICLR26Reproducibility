# DEBUG PROBE: judge environment + CPU scaling
import os, time
import numpy as np
import numba
from numba import prange

try:
    import psutil
except Exception:
    psutil = None


@numba.njit(fastmath=False)
def _nb_par(out, a, n):
    for i in prange(1, n - 2):
        out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2])


@numba.njit(parallel=True)
def _nb_copy(dst, src, n):
    for i in prange(n):
        dst[i] = src[i]


def _p(msg):
    print(msg, flush=True)


def fuse_stencil_through_transient(out, a, LEN_1D):
    n = int(LEN_1D)
    _p("AFFINITY=%d CPUINFO=%s CGROUP_MAX=%s" % (
        len(os.sched_getaffinity(0)),
        sum(1 for _ in open("/proc/cpuinfo") if "processor" in _),
        open("/sys/fs/cgroup/cpu.max").read().strip() if os.path.exists("/sys/fs/cgroup/cpu.max") else "?"))
    _p("PHYS=%s LOG=%s" % (
        psutil.cpu_count(logical=False) if psutil else "?",
        psutil.cpu_count(logical=True) if psutil else "?"))
    try:
        _p("NUMA: " + open("/sys/devices/system/node/online").read().strip())
    except Exception as e:
        _p("NUMA err %r" % e)

    if n < 4:
        return None
    # thread sweep
    for nt in [1, 4, 8, 16, 24, 32, 48, 64, 96, 192]:
        numba.set_num_threads(min(nt, len(os.sched_getaffinity(0))))
        # warm
        _nb_par(out, a, 100000)
        t0 = time.perf_counter()
        _nb_par(out, a, n)
        dt = time.perf_counter() - t0
        _p("PRANGE nthreads=%d t=%.1fms bw=%.0fGB/s" % (
            numba.get_num_threads(), dt * 1e3, (n * 16) / dt / 1e9))
    # fixed timing for chosen
    numba.set_num_threads(min(64, len(os.sched_getaffinity(0))))
    t0 = time.perf_counter(); _nb_par(out, a, n); tpar = time.perf_counter() - t0
    _p("PRANGE final: %.1fms eff=%.0fGB/s" % (tpar * 1e3, (n * 16) / tpar / 1e9))
    # parallel copy bandwidth
    dst = np.empty(n, np.float64)
    _nb_copy(dst, a, 100000)
    t0 = time.perf_counter(); _nb_copy(dst, a, n); tcopy = time.perf_counter() - t0
    _p("PARCOPY 64t: %.1fms eff=%.0fGB/s" % (tcopy * 1e3, (n * 16) / tcopy / 1e9))
    del dst
    # single-thread copy
    t0 = time.perf_counter(); np.copyto(out, a); t1c = time.perf_counter() - t0
    _p("NPYCOPY 1t: %.1fms eff=%.0fGB/s" % (t1c * 1e3, (n * 16) / t1c / 1e9))
    # torch pinned
    try:
        import torch
        torch.zeros(8, device='cuda')
        pa = torch.empty(n, dtype=torch.float64, pin_memory=True)
        po = torch.empty(n, dtype=torch.float64, pin_memory=True)
        pa.copy_(torch.from_numpy(a))
        torch.cuda.synchronize()
        t0 = time.perf_counter(); pa.copy_(torch.from_numpy(a)); torch.cuda.synchronize(); tA = time.perf_counter() - t0
        ga = pa.to('cuda', non_blocking=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter(); ga2 = pa.to('cuda', non_blocking=True); torch.cuda.synchronize(); tB = time.perf_counter() - t0
        po.copy_(ga2, non_blocking=True)
        t0 = time.perf_counter(); torch.cuda.synchronize(); tC = time.perf_counter() - t0
        _p("PINNED: cpu2pinned=%.1fms pinned2dev=%.1fms dev2pinned=%.1fms (=%.0f/%.0f/%.0f GB/s)" % (
            tA * 1e3, tB * 1e3, tC * 1e3, n * 8 / tA / 1e9, n * 8 / tB / 1e9, n * 8 / tC / 1e9))
        # gpu alloc cost
        t0 = time.perf_counter(); x = torch.empty(n, dtype=torch.float64, device='cuda'); del x; torch.cuda.synchronize(); _p("ALLOC %.0fGB: %.1fms" % (n*8/1e9, (time.perf_counter()-t0)*1e3))
    except Exception as e:
        _p("TORCH err %r" % e)
    return None
