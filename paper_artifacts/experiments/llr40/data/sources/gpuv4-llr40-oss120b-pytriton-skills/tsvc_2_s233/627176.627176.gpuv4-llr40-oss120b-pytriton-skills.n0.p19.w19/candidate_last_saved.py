import numpy as np

def s233(aa, bb, cc, LEN_2D):
    if LEN_2D <= 8:
        return None
    aa[8:, 8:] = aa[7, 8:] + np.cumsum(cc[8:, 8:], axis=0)
    bb[8:, 8:] = bb[8:, 7][:, None] + np.cumsum(cc[8:, 8:], axis=1)
    return None

