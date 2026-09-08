import numpy as np
import numba

@numba.njit
def compact_threshold_pack(src, weight, packed, out_count, LEN_1D):
    """Numba-accelerated stream compaction.
    Packs src[i] * weight[i] for src[i] > 0 into 'packed', preserving order.
    Stores the survivor count in out_count[0].
    """
    src_arr = src
    weight_arr = weight
    packed_arr = packed
    n = 0
    for i in range(LEN_1D):
        val = src_arr[i]
        if val > 0.0:
            packed_arr[n] = val * weight_arr[i]
            n += 1
    out_count[0] = n

