"""fuse_diamond: out[i] = (a[i]^2 + 1) * (a[i]^2 - 1), fp64.

Strategy: the kernel is a pure elementwise fuse; on this APU the GPU compute
is ~50x faster than the CPU loop but host<->device transfers run at ~58 GB/s
per GPU.  We split the array across all visible GPUs, pipeline H2D and D2H
on separate streams (both directions run concurrently), and run the fused
Triton kernel in place on device.  Small inputs stay on CPU (numba, serial or
multi-threaded) where launch overhead dominates.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")

import numpy as np
import torch
import triton
import triton.language as tl
import numba

# ---------------------------------------------------------------------------
# devices / streams / events (module import = untimed)
# ---------------------------------------------------------------------------
torch.cuda.init()
NGPU = torch.cuda.device_count()
STREAMS_A = [torch.cuda.Stream(device=d) for d in range(NGPU)]
STREAMS_B = [torch.cuda.Stream(device=d) for d in range(NGPU)]
DEVS = [f"cuda:{d}" for d in range(NGPU)]

# persistent per-GPU scratch buffers (grow as needed)
_SCRATCH = [None] * NGPU
_SCRATCH_N = [0] * NGPU


def _scratch(d, need):
    global _SCRATCH_N
    t = _SCRATCH[d]
    if t is None or t.numel() < need:
        t = torch.empty(need, device=DEVS[d], dtype=torch.float64)
        _SCRATCH[d] = t
        _SCRATCH_N[d] = t.numel()
    return t


# ---------------------------------------------------------------------------
# triton kernel: in-place fused diamond, bit-identical to the reference ops
# ---------------------------------------------------------------------------
BLOCK = 2048


@triton.jit
def _fd_ip(a_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(a_ptr + offs, mask=m)
    t = x * x
    tl.store(a_ptr + offs, (t + 1.0) * (t - 1.0), mask=m)


def _gpu_launch(d, dev_buf, s):
    _fd_ip[(triton.cdiv(s, BLOCK),)](
        dev_buf, s, BLOCK=BLOCK, num_warps=8, enable_fp_fusion=False
    )


# warm: compile + one launch per device (untimed)
def _warm():
    for d in range(NGPU):
        buf = _scratch(d, 2 * BLOCK)
        buf.fill_(1.5)
        torch.cuda.set_device(d)
        _gpu_launch(d, buf, 2 * BLOCK)
    torch.cuda.synchronize()


_warm()

# ---------------------------------------------------------------------------
# CPU fallbacks (numba, warmed at import)
# ---------------------------------------------------------------------------


@numba.njit(fastmath=False)
def _cpu_serial(out, a, n):
    for i in range(n):
        t = a[i] * a[i]
        out[i] = (t + 1.0) * (t - 1.0)


@numba.njit(parallel=True, fastmath=False)
def _cpu_mt(out, a, n):
    for i in numba.prange(n):
        t = a[i] * a[i]
        out[i] = (t + 1.0) * (t - 1.0)


def _warm_cpu():
    x = np.arange(1 << 14, dtype=np.float64)
    y = np.empty_like(x)
    _cpu_serial(y, x, x.size)
    _cpu_mt(y, x, x.size)


_warm_cpu()

# crossover thresholds (elements); tuned empirically
T_MT = 1 << 16      # >= this: prefer MT numba over serial
T_GPU = 1 << 21     # >= this and multi-GPU available: prefer GPU path


def _chunking(n):
    """return list of (offset, count): 4 sub-chunks per GPU, balanced."""
    nchunk = 4 * NGPU
    per = max(1, n // nchunk)
    counts = [per] * nchunk
    counts[-1] += n - per * nchunk
    chunks = []
    off = 0
    for c in counts:
        if c <= 0:
            break
        chunks.append((off, c))
        off += c
    return chunks


def fuse_diamond(out, a, LEN_1D):
    n = int(LEN_1D)
    if n <= 0:
        return None
    a = np.ascontiguousarray(a)
    out = np.ascontiguousarray(out)
    if a.dtype == np.float64 and out.dtype == np.float64 and NGPU and n >= T_GPU:
        chunks = _chunking(n)
        for d in range(NGPU):
            sA = STREAMS_A[d]
            sB = STREAMS_B[d]
            torch.cuda.set_device(d)
            for i in range(4 * d, 4 * d + 4):
                if i >= len(chunks):
                    break
                off, c = chunks[i]
                buf = _scratch(d, c)
                with torch.cuda.stream(sA):
                    buf[:c].copy_(torch.from_numpy(a[off:off + c]), non_blocking=True)
                    _gpu_launch(d, buf[:c], c)
                    ev = torch.cuda.Event()
                    ev.record(sA)
                sB.wait_event(ev)
                with torch.cuda.stream(sB):
                    torch.from_numpy(out[off:off + c]).copy_(buf[:c], non_blocking=True)
        for d in range(NGPU):
            torch.cuda.synchronize(d)
        return None
    if n >= T_MT:
        _cpu_mt(out, a, n)
    else:
        _cpu_serial(out, a, n)
    return None
