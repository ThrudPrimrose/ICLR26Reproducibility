import sys
import numpy as np

def s235(a, b, c, aa, bb, LEN_2D):
    out = []
    out.append("shapes: a %s b %s c %s aa %s bb %s LEN_2D=%r" % (
        tuple(a.shape), tuple(b.shape), tuple(c.shape), tuple(aa.shape), tuple(bb.shape), LEN_2D))
    out.append("dtypes: %s %s %s %s %s" % (a.dtype, b.dtype, c.dtype, aa.dtype, bb.dtype))
    out.append("contig: a %s b %s c %s aa %s bb %s" % (
        a.flags['C_CONTIGUOUS'], b.flags['C_CONTIGUOUS'], c.flags['C_CONTIGUOUS'],
        aa.flags['C_CONTIGUOUS'], bb.flags['C_CONTIGUOUS']))
    out.append("strides aa %s bb %s" % (aa.strides, bb.strides))
    out.append("a[:4] %s b[:4] %s c[:4] %s" % (a[:4], b[:4], c[:4]))
    out.append("aa[0, :4] %s bb[1, :4] %s" % (aa[0, :4], bb[1, :4]))
    sys.stdout.write("\n".join(out) + "\n")
    sys.stdout.flush()
    # perform the reference computation in place so outputs are valid
    for i in range(LEN_2D):
        a[i] = a[i] + b[i] * c[i]
        for j in range(1, LEN_2D):
            aa[j, i] = aa[j - 1, i] + bb[j, i] * a[i]
    return None
