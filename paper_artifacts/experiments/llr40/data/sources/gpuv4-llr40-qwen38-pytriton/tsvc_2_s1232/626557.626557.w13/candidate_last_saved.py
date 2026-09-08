import sys
import numpy as np

_seen = set()

def s1232(aa, bb, cc, LEN_2D, VLEN):
    key = (aa.shape, aa.strides, aa.dtype, type(LEN_2D), type(VLEN), repr(LEN_2D), repr(VLEN))
    if key not in _seen:
        _seen.add(key)
        print("PROBE shape", aa.shape, "strides", aa.strides, "dtype", aa.dtype,
              "LEN_2D", repr(LEN_2D), "VLEN", repr(VLEN),
              "bb flags", bb.flags.c_contiguous, cc.flags.c_contiguous,
              flush=True)
    L = int(LEN_2D)
    i = np.arange(L)
    m = i[:, None] >= i * VLEN
    aa[m] = bb[m] + cc[m]
