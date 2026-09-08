import numpy as np


def s318(a, result, inc, LEN_1D):
    s = a[0:LEN_1D * inc:inc]
    idx = int(np.argmax(np.abs(s)))
    result[0] = abs(float(a[idx * inc])) + float(idx)
    return None
