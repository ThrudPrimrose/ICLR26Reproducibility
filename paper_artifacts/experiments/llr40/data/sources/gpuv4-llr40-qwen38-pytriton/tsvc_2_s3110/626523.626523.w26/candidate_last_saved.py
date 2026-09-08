import sys
import numpy as np

def s3110(aa, bb, LEN_2D):
    print("AA shape", aa.shape, "strides", aa.strides, "dtype", aa.dtype, "C-contig", np.ascontiguousarray(aa) is aa or aa.flags['C_CONTIGUOUS'])
    print("BB shape", bb.shape, "dtype", bb.dtype, "LEN_2D", LEN_2D)
    sys.stdout.flush()
    idx = np.argmax(aa)
    n = aa.shape[0]
    i = idx // n
    j = idx % n
    bb[0, 0] = aa.flat[idx] + float(i) + float(j)
    return None

tsvc_2_s3110_fp64 = s3110
