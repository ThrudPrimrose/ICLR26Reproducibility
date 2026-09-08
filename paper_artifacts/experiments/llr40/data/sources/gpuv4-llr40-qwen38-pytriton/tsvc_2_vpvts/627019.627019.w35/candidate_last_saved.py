import os, sys, time, shutil, subprocess
import numpy as np

out = []

def which(names):
    return {n: shutil.which(n) for n in names}

out.append("tools " + str(which(["gcc", "g++", "clang", "icc", "cc"])))
out.append(f"affinity={len(os.sched_getaffinity(0))}")

C_SRC = r'''
#include <omp.h>
#include <stdint.h>
void axpy_omp(double *a, const double *b, int64_t n, double S, int nthreads) {
    omp_set_num_threads(nthreads);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) a[i] += b[i] * S;
}
void axpy_unroll8(double *a, const double *b, int64_t n, double S, int nthreads) {
    omp_set_num_threads(nthreads);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i += 8) {
        int64_t j = i + 7;
        if (j >= n) j = n - 1;
        #pragma GCC ivdep
        for (int k = 0; k < 8; ++k) a[i + k] += b[i + k] * S;
    }
}
'''

_ext = None
def _load_ext():
    global _ext
    try:
        src = "/tmp/vpvts_probe.c"
        so = "/tmp/libvpvts_probe.so"
        if not (os.path.exists(so) and os.path.getmtime(so) > os.path.getmtime(src) if os.path.exists(src) else True):
            with open(src, "w") as f:
                f.write(C_SRC)
            r = subprocess.run(
                ["gcc", "-O3", "-march=native", "-ffast-math", "-fopenmp", "-fPIC", "-shared", src, "-o", so],
                capture_output=True, text=True, timeout=180)
            out.append(f"gcc rc={r.returncode} err={r.stderr[:300]}")
            if r.returncode != 0:
                return
        import ctypes
        _ext = ctypes.CDLL(so)
        for fn in ("axpy_omp", "axpy_unroll8"):
            f = getattr(_ext, fn)
            f.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                          ctypes.c_int64, ctypes.c_double, ctypes.c_int]
            f.restype = None
        out.append("ext loaded")
    except Exception as e:
        out.append(f"c-err {e!r}")

_load_ext()

def _bench(a, b, n, S):
    m = min(60_000_000, n)
    if m < 10_000_000:
        m = 0
    if m == 0:
        return
    a2 = np.array(a[:m]); b2 = np.array(b[:m])
    def bench(f, *args, reps=3):
        f(*args); ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); f(*args); ts.append(time.perf_counter() - t0)
        return ts
    def show(name, f, *args):
        ts = bench(f, *args)
        out.append(f"{name:14s} " + " ".join(f"{t*1e3:7.2f}ms {24*m/t/1e9:7.1f}GB/s" for t in ts))
    if _ext is not None:
        pa = a2.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        pb = b2.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        show("c_omp_1", _ext.axpy_omp, pa, pb, m, S, 1)
        show("c_omp_24", _ext.axpy_omp, pa, pb, m, S, 24)
        show("c_unr8_24", _ext.axpy_unroll8, pa, pb, m, S, 24)
    import importlib.util
    spec = importlib.util.spec_from_file_location("vmod", "/shared/agent-35/vmod_impl.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    show("nb1", mod._vpvts1, a2, b2, m, S)
    show("nbP24", mod._vpvtsP, a2, b2, m, S)

def vpvts(a, b, LEN_1D, S):
    if not getattr(vpvts, "_probed", False):
        vpvts._probed = True
        _bench(a, b, LEN_1D, S)
        sys.stdout.write("\n".join(out) + "\n")
        sys.stdout.flush()
    import importlib.util
    spec = importlib.util.spec_from_file_location("vmod", "/shared/agent-35/vmod_impl.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.vpvts(a, b, LEN_1D, S)
