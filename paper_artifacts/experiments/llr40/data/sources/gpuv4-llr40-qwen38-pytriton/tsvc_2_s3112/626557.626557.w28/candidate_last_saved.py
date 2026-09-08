"""TSVC s3112 (inclusive prefix scan), optimized for the python arm.

Strategy
--------
b[i] = sum_{j<=i} a[j].  The reference is a serial chain; we re-associate the
addition order (parallel scan) and run it on the available AMD GPUs:

  1. split the array over G GPUs, DMA a -> device (hipMemcpy, ~58 GB/s)
  2. triton kernel 1: per-block chunk sums
  3. tiny D2H, serial numpy prefix of chunk sums, tiny H2D
  4. triton kernel 2: per-block inclusive cumsum + chunk offset -> b
  5. DMA device -> b directly (hipMemcpy)

Everything (imports, JIT compiles, DMA-path warmup, buffer pools) is done at
import time; the timed call only does the pipeline above.  Any GPU failure
falls back to a bitwise-exact serial numba scan.
"""

import glob
import os
import ctypes

import numpy as np
from numba import njit


# ---------------------------------------------------------------- serial path
@njit
def _scan_serial(a, b, n):
    s = 0.0
    for i in range(n):
        s += a[i]
        b[i] = s


def _serial(a, b, n):
    _scan_serial(a, b, n)


# warm the JIT now (untimed)
_d = np.ones(64)
_serial(_d, _d, 64)

# ------------------------------------------------------------- GPU machinery
BLOCK = 4096
_STATE = {"ok": False}


def _init_gpu():
    st = _STATE
    try:
        import torch
        import triton
        import triton.language as tl

        if not torch.cuda.is_available():
            return
        G = min(int(torch.cuda.device_count()), 4)
        if G < 1:
            return

        hip = None
        for pat in (
            "/opt/rocm*/lib/libamdhip64.so*",
            "/opt/rocm*/lib64/libamdhip64.so*",
            "/usr/lib/libamdhip64.so*",
            "/usr/lib64/libamdhip64.so*",
        ):
            for p in sorted(glob.glob(pat)):
                try:
                    hip = ctypes.CDLL(p)
                    break
                except OSError:
                    continue
            if hip is not None:
                break
        if hip is None:
            return
        hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_size_t, ctypes.c_int]
        hip.hipMemcpy.restype = ctypes.c_int
        hip.hipSetDevice.argtypes = [ctypes.c_int]
        hip.hipSetDevice.restype = ctypes.c_int

        @triton.jit(do_not_specialize=["n"])
        def k1(a_ptr, p_ptr, n, BLK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLK + tl.arange(0, BLK)
            mask = offs < n
            x = tl.load(a_ptr + offs, mask=mask, other=0.0)
            tl.store(p_ptr + pid, tl.sum(x, axis=0))

        @triton.jit(do_not_specialize=["n"])
        def k2(a_ptr, q_ptr, b_ptr, n, BLK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLK + tl.arange(0, BLK)
            mask = offs < n
            x = tl.load(a_ptr + offs, mask=mask, other=0.0)
            c = tl.cumsum(x, axis=0)
            start = tl.load(q_ptr + pid)
            tl.store(b_ptr + offs, c + start, mask=mask)

        devs = [torch.device("cuda", i) for i in range(G)]
        SEG = 200 * 10**6                      # max elements per GPU
        BLKMAX = SEG // BLOCK + 2
        ta, tb, pdev, qdev = [], [], [], []
        phost, qhost = [], []
        for g in range(G):
            torch.cuda.set_device(devs[g])
            ta.append(torch.empty(SEG, dtype=torch.float64, device=devs[g]))
            tb.append(torch.empty(SEG, dtype=torch.float64, device=devs[g]))
            pdev.append(torch.empty(BLKMAX, dtype=torch.float64, device=devs[g]))
            qdev.append(torch.empty(BLKMAX, dtype=torch.float64, device=devs[g]))
            phost.append(np.empty(BLKMAX, dtype=np.float64))
            qhost.append(np.empty(BLKMAX, dtype=np.float64))

        # --- warmup: compile kernels on every device + warm DMA paths ------
        hip.hipMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_size_t, ctypes.c_int,
                                       ctypes.c_void_p]
        hip.hipMemcpyAsync.restype = ctypes.c_int
        small = 2 * BLOCK

        def _w1(g):
            torch.cuda.set_device(devs[g])
            hip.hipSetDevice(g)
            a_np = np.random.default_rng(g).standard_normal(small)
            b_np = np.empty(small)
            hip.hipMemcpyAsync(ctypes.c_void_p(ta[g].data_ptr()),
                               ctypes.c_void_p(a_np.ctypes.data),
                               a_np.nbytes, 1, ctypes.c_void_p(0))
            k1[(2,)](ta[g], pdev[g], small, BLK=BLOCK)
            torch.cuda.synchronize()
            hip.hipMemcpyAsync(ctypes.c_void_p(phost[g].ctypes.data),
                               ctypes.c_void_p(pdev[g].data_ptr()),
                               2 * 8, 2, ctypes.c_void_p(0))
            torch.cuda.synchronize()
            qhost[g][:2] = [0.0, phost[g][0]]
            hip.hipMemcpyAsync(ctypes.c_void_p(qdev[g].data_ptr()),
                               ctypes.c_void_p(qhost[g].ctypes.data),
                               2 * 8, 1, ctypes.c_void_p(0))
            k2[(2,)](ta[g], qdev[g], tb[g], small, BLK=BLOCK)
            torch.cuda.synchronize()
            hip.hipMemcpyAsync(ctypes.c_void_p(b_np.ctypes.data),
                               ctypes.c_void_p(tb[g].data_ptr()),
                               b_np.nbytes, 2, ctypes.c_void_p(0))
            torch.cuda.synchronize()
            assert np.allclose(b_np, np.cumsum(a_np), rtol=1e-9,
                               atol=1e-9), "warmup mismatch g=%d" % g

        for g in range(G):
            _w1(g)

        st.update(ok=True, G=G, devs=devs, hip=hip, ta=ta, tb=tb,
                  pdev=pdev, qdev=qdev, phost=phost, qhost=qhost,
                  k1=k1, k2=k2, torch=torch, SEG=SEG, BLKMAX=BLKMAX)
    except Exception:
        st["ok"] = False


_init_gpu()


def _gpu_scan(a, b, n):
    st = _STATE
    torch, hip = st["torch"], st["hip"]
    G = st["G"]
    mca = hip.hipMemcpyAsync
    zero = ctypes.c_void_p(0)

    # segment plan
    base = n // G
    seg = [base + (1 if g < n % G else 0) for g in range(G)]
    off = [0] * G
    for g in range(1, G):
        off[g] = off[g - 1] + seg[g - 1]
    nblk = [int((seg[g] + BLOCK - 1) // BLOCK) for g in range(G)]
    tot = sum(nblk)
    boff = [0] * G
    for g in range(1, G):
        boff[g] = boff[g - 1] + nblk[g - 1]

    a_ptr = a.ctypes.data
    b_ptr = b.ctypes.data
    ap = ctypes.c_void_p

    # ---- phase 1: async H2D + chunk sums + async D2H of chunk sums --------
    for g in range(G):
        torch.cuda.set_device(st["devs"][g])
        hip.hipSetDevice(g)
        mca(ap(st["ta"][g].data_ptr()), ap(a_ptr + off[g] * 8),
            seg[g] * 8, 1, zero)
        st["k1"][(nblk[g],)](st["ta"][g], st["pdev"][g], seg[g], BLK=BLOCK)
        mca(ap(st["phost"][g].ctypes.data), ap(st["pdev"][g].data_ptr()),
            nblk[g] * 8, 2, zero)
    torch.cuda.synchronize()

    # serial prefix over all chunk sums (numpy cumsum: plain serial C loop)
    allp = np.empty(tot)
    for g in range(G):
        allp[boff[g]:boff[g] + nblk[g]] = st["phost"][g][:nblk[g]]
    cs = np.cumsum(allp)
    pb = np.empty(tot)
    pb[0] = 0.0
    if tot > 1:
        pb[1:] = cs[:-1]
    for g in range(G):
        st["qhost"][g][:nblk[g]] = pb[boff[g]:boff[g] + nblk[g]]

    # ---- phase 2: async H2D of offsets + scan + async D2H into b ----------
    for g in range(G):
        torch.cuda.set_device(st["devs"][g])
        hip.hipSetDevice(g)
        mca(ap(st["qdev"][g].data_ptr()), ap(st["qhost"][g].ctypes.data),
            nblk[g] * 8, 1, zero)
        st["k2"][(nblk[g],)](st["ta"][g], st["qdev"][g], st["tb"][g],
                             seg[g], BLK=BLOCK)
        mca(ap(b_ptr + off[g] * 8), ap(st["tb"][g].data_ptr()),
            seg[g] * 8, 2, zero)
    torch.cuda.synchronize()


def _fallback(a, b, n):
    if isinstance(a, np.ndarray) and a.dtype == np.float64 \
            and a.flags.c_contiguous and isinstance(b, np.ndarray) \
            and b.dtype == np.float64 and b.flags.c_contiguous:
        _scan_serial(a, b, n)
    else:
        ac = np.ascontiguousarray(a)
        tmp = np.cumsum(ac[:n])
        if isinstance(b, np.ndarray):
            b[:n] = tmp
        else:
            b[...] = tmp


def s3112(a, b, LEN_1D):
    n = int(LEN_1D)
    if n <= 0:
        return None
    use_gpu = (
        _STATE["ok"]
        and isinstance(a, np.ndarray) and a.dtype == np.float64
        and a.flags.c_contiguous
        and isinstance(b, np.ndarray) and b.dtype == np.float64
        and b.flags.c_contiguous
        and n >= 8 * BLOCK
        and n <= _STATE.get("G", 0) * _STATE.get("SEG", 0)
    )
    if not use_gpu:
        _fallback(a, b, n)
        return None
    try:
        _gpu_scan(a, b, n)
    except Exception:
        _fallback(a, b, n)
    return None
