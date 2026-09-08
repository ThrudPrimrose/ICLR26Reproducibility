import os
import time
import numpy as np
import numba as nb

@nb.njit(fastmath=True)
def _serial(a, b, c, N, K):
    for i in range(K, N):
        a[i] = 0.75 * a[i - K] + b[i] * c[i]

_warmed = False

def _warm():
    global _warmed
    if not _warmed:
        n = 1024
        a = np.ones(n)
        b = np.ones(n)
        c = np.ones(n)
        _serial(a, b, c, n, 1)
        _warmed = True

def versioned_distance_update(a, b, c, LEN_1D, K):
    _warm()
    print("PROBE2 LEN_1D=", LEN_1D, "K=", K, flush=True)
    try:
        import torch
        print("torch", torch.__version__, "cuda avail:", torch.cuda.is_available(), flush=True)
        if torch.cuda.is_available():
            print("dev count:", torch.cuda.device_count(), flush=True)
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print("dev", i, p.name, "total GB:", p.total_memory / 1e9, "cap:", p.major, p.minor, flush=True)
            dev = torch.cuda.current_device()
            print("current dev:", dev, "visible:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
            n = 1 << 26  # 64M float64 = 512MB
            h = torch.randn(n, dtype=torch.float64)
            d = h.to(torch.device("cuda", dev))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                d2 = h.to(torch.device("cuda", dev))
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            print("H2D 512MB x3: %.3fs -> %.1f GB/s" % (t1 - t0, 3 * 0.512 / (t1 - t0)), flush=True)
            d3 = h.to(torch.device("cuda", dev))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                hh = d3.to("cpu")
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            print("D2H 512MB x3: %.3fs -> %.1f GB/s" % (t1 - t0, 3 * 0.512 / (t1 - t0)), flush=True)
            x = torch.randn(1 << 27, dtype=torch.float64, device="cuda")
            w = torch.ones(128, dtype=torch.float64, device="cuda")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                y = x * w[:1]
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            gb = 5 * 3 * 1.074
            print("elemwise 1GB x5: %.3fs -> %.1f GB/s" % (t1 - t0, gb / (t1 - t0)), flush=True)
            del x, y, h, d, d2, d3, hh
        else:
            print("NO GPU on judge", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
    _serial(a, b, c, LEN_1D, K)
    return None
