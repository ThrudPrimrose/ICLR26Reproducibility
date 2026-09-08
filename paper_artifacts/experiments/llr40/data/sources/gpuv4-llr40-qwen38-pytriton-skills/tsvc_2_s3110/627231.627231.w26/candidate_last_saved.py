import numpy as np

def s3110(aa, bb, LEN_2D):
    print("PROBE", (tuple(aa.shape), aa.dtype, aa.strides, tuple(bb.shape), bb.dtype, LEN_2D,
          "contig=", aa.flags['C_CONTIGUOUS']), flush=True)
    bb[0, 0] = float(aa.max())
