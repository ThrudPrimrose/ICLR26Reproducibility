import numpy as np

def s3110(aa, bb, LEN_2D):
    idx = np.argmax(aa)
    bb[0, 0] = aa.flat[idx] + float(idx // LEN_2D) + float(idx % LEN_2D)
    return None
