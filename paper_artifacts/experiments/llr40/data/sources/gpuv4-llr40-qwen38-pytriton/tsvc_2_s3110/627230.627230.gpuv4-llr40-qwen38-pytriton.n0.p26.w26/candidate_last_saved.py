"""TSVC tsvc_2 s3110 -- first-occurrence 2D argmax + coordinate checksum.

The reference performs a row-major strict-> scan: the output is the max value
plus the (i, j) of the FIRST occurrence.  np.argmax / torch.argmax both
return the first occurrence, so a plain argmax is exactly right.

Strategy (all expensive setup at import time, nothing here is compiled):
  * small inputs : single-threaded NumPy argmax (fastest at cache sizes)
  * large inputs : torch argmax on the MI300 GPU (H2D transfer + GPU reduce);
    the ROCm context and allocator are warmed up at import so the timed
    section contains no initialisation.
In-place ABI: writes bb[0, 0], returns None.
"""
import numpy as _np

__all__ = ["s3110"]

_TORCH = None
_DEV = None
_CROSSOVER = 1500000  # elements at/above which the GPU path wins on this cluster

def _setup_gpu():
    global _TORCH, _DEV
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    try:
        dev = torch.device("cuda")
        x = torch.arange(8, device=dev, dtype=torch.float64)
        int(x.argmax().item())          # context + allocator warm-up
        y = torch.randn(1 << 16)
        int(y.to(dev).argmax().item())  # warm the H2D + reduce + sync path
        del x, y
        torch.cuda.synchronize()
        _TORCH = torch
        _DEV = dev
    except Exception:
        _TORCH = None
        _DEV = None

_setup_gpu()


def s3110(aa, bb, LEN_2D):
    flat = _np.ascontiguousarray(aa, dtype=_np.float64).reshape(-1)
    n = flat.size
    if _TORCH is not None and n >= _CROSSOVER:
        k = int(_TORCH.from_numpy(flat).to(_DEV).argmax().item())
    else:
        k = int(flat.argmax())
    maxv = float(flat[k])
    i, j = divmod(k, LEN_2D)
    bb[0, 0] = maxv + float(i) + float(j)
    return None
