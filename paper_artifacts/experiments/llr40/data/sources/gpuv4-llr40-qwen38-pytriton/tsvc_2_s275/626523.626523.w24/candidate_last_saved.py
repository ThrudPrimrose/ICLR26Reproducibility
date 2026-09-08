import os, sys, time
import numpy as np
import numba

@numba.njit(fastmath=True)
def serial(aa, bb, cc, n):
    for i in range(n):
        if aa[0, i] > 0.0:
            for j in range(1, n):
                aa[j, i] = aa[j - 1, i] + bb[j, i] * cc[j, i]

@numba.njit(parallel=True, fastmath=True)
def row_add(x, y, out):
    m, k = x.shape
    for r in numba.prange(m):
        for c in range(k):
            out[r, c] = x[r, c] + y[r, c]

def s275(aa, bb, cc, LEN_2D):
    n = LEN_2D
    try:
        aff = os.sched_getaffinity(0)
        print("AFFINITY", len(aff), sorted(aff)[:6], "...")
    except Exception as e:
        print("AFF err", e)
    print("LOADAVG", os.getloadavg())
    a = aa.copy()
    numba.set_num_threads(1)
    reps = []
    for r in range(3):
        t0 = time.perf_counter(); serial(a, bb, cc, n); t1 = time.perf_counter()
        reps.append((t1 - t0) * 1e3)
    print("SERIAL1_MS", reps)
    numba.set_num_threads(24)
    scratch = np.empty_like(bb)
    row_add(bb, cc, scratch)
    reps = []
    for r in range(3):
        t0 = time.perf_counter(); row_add(bb, cc, scratch); t1 = time.perf_counter()
        reps.append((t1 - t0) * 1e3)
    print("ROWADD_T24_MS", reps, "GBPS_min", 3 * aa.nbytes / min(reps) / 1e6)
    try:
        import triton
        import triton.language as tl
        print("TRITON_VERSION", triton.__version__)
        @triton.jit
        def k_add(xp, yp, op, N, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            m = offs < N
            tl.store(op + offs, tl.load(xp + offs, mask=m) + tl.load(yp + offs, mask=m), mask=m)
        N = 4096
        x = np.ones(N); y = np.ones(N) * 2; o = np.zeros(N)
        t0 = time.perf_counter()
        k_add[(N // 256,)](x, y, o, N, BLOCK=256)
        t1 = time.perf_counter()
        print("TRITON_SMALL_OK", bool((o == 3).all()), "compile_s", (t1 - t0))
        big = np.empty(n * n)
        k_add[(n * n // 1024,)](bb, cc, big, n * n, BLOCK=1024)
        reps = []
        for r in range(3):
            t0 = time.perf_counter()
            k_add[(n * n // 1024,)](bb, cc, big, n * n, BLOCK=1024)
            t1 = time.perf_counter()
            reps.append((t1 - t0) * 1e3)
        print("TRITON_BIG_MS", reps, "GBPS_min", 3 * aa.nbytes / min(reps) / 1e6)
    except Exception as e:
        import traceback; traceback.print_exc()
        print("TRITON_FAIL", type(e).__name__, str(e)[:300])
    try:
        import torch
        print("TORCH", torch.__version__, "cuda", torch.cuda.is_available(),
              torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    except Exception as e:
        print("TORCH_FAIL", type(e).__name__, str(e)[:200])
    sys.stdout.flush()
    return None
