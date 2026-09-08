"""Optimized tsvc_2_s231: column-wise recurrence aa[j,i] = aa[j-1,i] + bb[j,i].

Closed form: aa_new[j,i] = aa[0,i] + sum_{k=1..j} bb[k,i]
            = cumsum_axis0(bb)[j,i] + (aa[0] - bb[0])[i]
"""
import numpy as np


def s231(aa, bb, LEN_2D):
    n = int(LEN_2D)
    d = aa[0] - bb[0]
    np.cumsum(bb, axis=0, out=aa)
    aa += d
    return None
