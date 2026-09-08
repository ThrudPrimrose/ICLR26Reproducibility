import numpy as np


def argmax_with_index(a, out_value, out_index, LEN_1D):
    i = np.argmax(a)
    out_value[0] = a[i]
    out_index[0] = i
