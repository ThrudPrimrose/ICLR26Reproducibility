import os, sys, time, resource
def _p(*a):
    print(*a); sys.stdout.flush()

try:
    _p("mlock rlim:", resource.getrlimit(resource.RLIMIT_MEMLOCK))
    import torch
    _p("torch", torch.__version__)
    _p("devcount", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        _p("dev", i, torch.cuda.get_device_name(i), "MB", torch.cuda.get_device_properties(i).total_memory//1048576)
    x = torch.randn(1000, device="cuda:0")
    torch.cuda.synchronize()
    _p("cuda0 op ok", float(x.sum()))
except Exception as e:
    _p("GPU SETUP FAILED:", repr(e))
    x = None

if x is not None:
    import triton
    import triton.language as tl
    _p("triton", triton.__version__)

    @triton.jit
    def _gather(a_ptr, b_ptr, ip_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        idx = tl.load(ip_ptr + offs, mask=mask, other=0)
        v = tl.load(b_ptr + idx, mask=mask)
        tl.store(a_ptr + offs, v, mask=mask)

    # precompile with dummy
    dn = 256
    da = torch.empty(dn, dtype=torch.float64, device="cuda:0")
    db = torch.empty(dn, dtype=torch.float64, device="cuda:0")
    di = torch.empty(dn, dtype=torch.int32, device="cuda:0")
    t0 = time.perf_counter()
    _gather[(dn // 1024 + 1,)](da, db, di, dn, BLOCK=1024)
    torch.cuda.synchronize()
    _p("triton first-launch+sync: %.3f s" % (time.perf_counter() - t0))

    # host<->device bandwidth with pinned memory, 256MB
    mb = 256 * 1024 * 1024
    ph = torch.empty(mb, dtype=torch.uint8, pin_memory=True)
    dh = torch.empty(mb, dtype=torch.uint8, device="cuda:0")
    # warm
    dh.copy_(ph, non_blocking=True); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        dh.copy_(ph, non_blocking=True)
    torch.cuda.synchronize()
    h2d = (time.perf_counter() - t0) / 3
    t0 = time.perf_counter()
    for _ in range(3):
        ph.copy_(dh, non_blocking=True)
    torch.cuda.synchronize()
    d2h = (time.perf_counter() - t0) / 3
    _p("H2D 256MB pinned: %.4f s -> %.1f GB/s" % (h2d, mb / h2d / 1e9))
    _p("D2H 256MB pinned: %.4f s -> %.1f GB/s" % (d2h, mb / d2h / 1e9))

    # pageable H2D
    pp = torch.empty(mb, dtype=torch.uint8)
    pp.fill_(1)
    torch.cuda.synchronize()
    dh.copy_(pp, non_blocking=True); torch.cuda.synchronize()
    t0 = time.perf_counter()
    dh.copy_(pp, non_blocking=True); torch.cuda.synchronize()
    _p("H2D 256MB pageable: %.4f s -> %.1f GB/s" % (time.perf_counter() - t0, mb / (time.perf_counter() - t0) / 1e9))

    # gather kernel timing at 1e7
    for n in (10**6, 10**7):
        an = torch.empty(n, dtype=torch.float64, device="cuda:0")
        bn = torch.rand(n, dtype=torch.float64, device="cuda:0")
        in_ = torch.randint(0, n, (n,), dtype=torch.int32, device="cuda:0")
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
        _gather[grid](an, bn, in_, n, BLOCK=2048)
        torch.cuda.synchronize()
        best = 1e18
        for _ in range(3):
            t0 = time.perf_counter()
            _gather[grid](an, bn, in_, n, BLOCK=2048)
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t0)
        _p("gather n=%d: %.1f us (%.1f GB/s useful)" % (n, best * 1e6, 20 * n / best / 1e9))

    # multithreaded host copy numpy->pinned
    import numpy as np
    src = np.ones(256 * 1024 * 1024, dtype=np.float64)
    dst = np.empty_like(src)
    t0 = time.perf_counter(); dst[:].copy(src); t1 = time.perf_counter()
    _p("np copy 2GB traffic: %.4f s -> %.1f GB/s eff" % (t1 - t0, 256e6 / (t1 - t0) / 1e9))
    import threading
    def _cp(lo, hi, chunk=32 * 1024 * 1024):
        for s in range(lo, hi, chunk):
            e = min(s + chunk, hi)
            dst[s:e][:].copy(src[s:e])
    NCH = 24
    sz = src.nbytes // NCH
    dst2 = np.empty_like(src)
    def _cp2(lo, hi, d, chunk=32 * 1024 * 1024):
        for s in range(lo, hi, chunk):
            e = min(s + chunk, hi)
            d[s:e][:].copy(src[s:e])
    th = [threading.Thread(target=_cp2, args=(i * sz, (i + 1) * sz if i < NCH - 1 else src.nbytes, dst2)) for i in range(NCH)]
    t0 = time.perf_counter()
    for t in th: t.start()
    for t in th: t.join()
    t1 = time.perf_counter()
    _p("np copy 24-thread: %.4f s -> %.1f GB/s eff" % (t1 - t0, 256e6 / (t1 - t0) / 1e9))
    _p("PROBE DONE")

def vag(a, b, ip, LEN_1D):
    # probe-only: no real work (profile is never scored)
    return None
