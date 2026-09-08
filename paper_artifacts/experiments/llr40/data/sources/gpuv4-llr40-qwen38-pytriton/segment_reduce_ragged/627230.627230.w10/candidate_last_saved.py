import numpy as np

def segment_reduce_ragged(row_ptr, val, w, out, NSEG):
    if NSEG:
        idx = row_ptr[:-1]
        tot = val.size
        if idx[-1] >= tot:
            idx = idx.copy()
            idx[-1] = tot - 1
        np.add.reduceat(val * w, idx, out=out)
        nz = row_ptr[:-1] != row_ptr[1:]
        if not nz.all():
            out[~nz] = 0.0
    return None
