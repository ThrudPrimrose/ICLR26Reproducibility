import numpy as np


def s119(aa, bb, LEN_2D):
    for i in range(1, LEN_2D):
        aa[i, 1:] = aa[i - 1, :-1] + bb[i, 1:]
