"""compact_threshold_pack: NumPy stream compaction (v1)."""
import numpy as np


def compact_threshold_pack(src, weight, packed, out_count, LEN_1D):
    mask = src > 0.0
    n = int(np.count_nonzero(mask))
    if n:
        prod = src * weight
        packed[:n] = prod[mask]
    out_count[0] = n
