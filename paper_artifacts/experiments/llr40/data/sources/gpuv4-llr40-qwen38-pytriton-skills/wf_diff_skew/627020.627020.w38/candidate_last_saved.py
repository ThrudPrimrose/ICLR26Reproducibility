import numpy as np
import triton
import triton.language as tl

_GPU_OK = False
_dev_cache = {}
BLOCK = 512
NUM_W = 4
SMALL_N = 256


@triton.jit
def _wf_kernel(p, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n - 1
    for i in range(1, n):
        cur = tl.load(p + i * n + offs, mask=mask, other=0.0)
        p0 = tl.load(p + (i - 1) * n + offs, mask=mask, other=0.0)
        p1 = tl.load(p + (i - 1) * n + offs + 1, mask=mask, other=0.0)
        tl.store(p + i * n + offs, cur + p0 + p1, mask=mask)


def _numpy_run(a, n):
    for i in range(1, n):
        a[i, :-1] += a[i - 1, :-1] + a[i - 1, 1:]


def _try_warm():
    global _GPU_OK
    if _GPU_OK:
        return
    try:
        import torch
        torch.zeros(4, device="cuda")
        torch.cuda.synchronize()
        for n0 in (64, 997):
            d = torch.zeros(n0, n0, dtype=torch.float64, device="cuda")
            _wf_kernel[(triton.cdiv(n0 - 1, BLOCK),)](d, n0, BLOCK=BLOCK, num_warps=NUM_W)
            h = torch.empty(n0, n0, dtype=torch.float64)
            h.copy_(d)
            torch.cuda.synchronize()
        _GPU_OK = True
    except Exception:
        _GPU_OK = False


_try_warm()


def wf_diff_skew(a, LEN_2D):
    n = int(LEN_2D)
    if n < 2:
        return None
    if not (_GPU_OK or _try_warm()):
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a)
        _numpy_run(a, n)
        return None
    if a.dtype not in (np.float64, np.float32) or n < SMALL_N:
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a)
        _numpy_run(a, n)
        return None
    import torch
    src = a if a.flags.c_contiguous else np.ascontiguousarray(a)
    key = (n, a.dtype)
    dev = _dev_cache.get(key)
    if dev is None:
        dt = torch.float64 if a.dtype is np.float64 else torch.float32
        dev = torch.empty(n, n, dtype=dt, device="cuda")
        _dev_cache[key] = dev
    dev.copy_(torch.from_numpy(src))
    _wf_kernel[(triton.cdiv(n - 1, BLOCK),)](dev, n, BLOCK=BLOCK, num_warps=NUM_W)
    torch.from_numpy(src).copy_(dev)
    if src is not a:
        np.copyto(a, src)
    return None
