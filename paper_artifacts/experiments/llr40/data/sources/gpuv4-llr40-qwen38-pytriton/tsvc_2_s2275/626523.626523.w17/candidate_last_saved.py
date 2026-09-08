import sys
import numpy as np

def s2275(a, b, c, d, aa, bb, cc, LEN_2D):
    print('numpy', np.__version__)
    for name, x in (('a', a), ('aa', aa), ('bb', bb), ('cc', cc)):
        ax = np.ascontiguousarray(x, dtype=np.float64)
        print(name, x.shape, x.dtype, 'C=', x.flags['C_CONTIGUOUS'], 'W=', x.flags['WRITEABLE'],
              'is_same=', ax is x, 'strides', x.strides)
    sys.stdout.flush()
    n = int(LEN_2D)
    a2 = np.ascontiguousarray(a, dtype=np.float64)
    b2 = np.ascontiguousarray(b, dtype=np.float64)
    c2 = np.ascontiguousarray(c, dtype=np.float64)
    d2 = np.ascontiguousarray(d, dtype=np.float64)
    aa2 = np.ascontiguousarray(aa, dtype=np.float64)
    bb2 = np.ascontiguousarray(bb, dtype=np.float64)
    cc2 = np.ascontiguousarray(cc, dtype=np.float64)
    a[...] = b2 + c2 * d2
    aa[...] = (aa2.reshape(n * n) + bb2.reshape(n * n) * cc2.reshape(n * n)).reshape(aa.shape)
    return None
