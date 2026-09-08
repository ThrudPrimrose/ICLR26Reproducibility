import os, sys, subprocess, time
import numpy as np

def _grab(cmd, t=30):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t)
        return (r.stdout + r.stderr).strip()[:1500]
    except Exception as e:
        return "ERR %r" % e

_GPU_LINES = []
_TG = None
_TRITON_OK = False
try:
    import torch
    _TORCH = torch
except Exception:
    _TORCH = None
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _gather_k(a_ptr, b_ptr, ip_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        idx = tl.load(ip_ptr + offs, mask=mask)
        vals = tl.load(b_ptr + idx, mask=mask)
        tl.store(a_ptr + offs, vals, mask=mask)

    def _tg(a, b, ip, n):
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _gather_k[grid](a, b, ip, n, BLOCK=BLOCK, num_warps=8)

    # warm: tiny compile
    _n = 4096
    _a = np.zeros(_n, np.float64)
    _b = np.random.rand(_n)
    _ip = (np.random.randint(0, _n, _n)).astype(np.int32)
    _tg(_a, _b, _ip, _n)
    try:
        torch.cuda.synchronize() if _TORCH is not None else triton.runtime.driver.active.context.synchronize()
    except Exception:
        pass
    _TRITON_OK = np.array_equal(_a, _b[_ip])
    _TG = _tg
except Exception as e:
    _TRITON_OK = False
    _TERR = repr(e)

def vag(a, b, ip, LEN_1D):
    n = int(LEN_1D)
    L = []
    if _TORCH is None:
        L.append("no torch")
    else:
        L.append("torch %s cuda_avail=%s ndev=%s" %
                 (_TORCH.__version__,
                  _TORCH.cuda.is_available(),
                  _TORCH.cuda.device_count() if _TORCH.cuda.is_available() else 0))
        if _TORCH.cuda.is_available():
            L.append("dev0: %s" % str(_TORCH.cuda.get_device_name(0)))
    L.append("triton_ok: %s" % _TRITON_OK)
    L.append("rocm-smi:\n" + _grab("rocm-smi --showproductname --showmeminfo vram 2>&1 | head -20"))
    L.append("lspci:\n" + _grab("lspci 2>/dev/null | grep -i -E 'vga|display|3d' | head -5"))
    if _TRITON_OK and n > 1000:
        t0 = time.perf_counter()
        _TG(a, b, ip, n)
        try:
            triton.runtime.driver.active.context.synchronize()
        except Exception:
            pass
        t1 = time.perf_counter()
        L.append("triton big: %.1f ms  correct=%s" %
                 ((t1-t0)*1e3, np.array_equal(a, b[ip])))
        t0 = time.perf_counter()
        _TG(a, b, ip, n)
        triton.runtime.driver.active.context.synchronize()
        t1 = time.perf_counter()
        L.append("triton big2: %.1f ms" % ((t1-t0)*1e3))
    else:
        a[:n] = b[ip]
    sys.stdout.write("\n".join(L))
    sys.stdout.flush()
    return None
