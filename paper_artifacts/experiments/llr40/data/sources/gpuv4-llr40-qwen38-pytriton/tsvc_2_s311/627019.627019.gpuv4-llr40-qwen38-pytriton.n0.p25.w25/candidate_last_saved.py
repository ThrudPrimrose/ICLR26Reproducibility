import numpy as np

def s311(a, sum_out, LEN_1D):
    print("LEN_1D:", LEN_1D, "shape:", getattr(a, "shape", None), "dtype:", getattr(a, "dtype", None), "outshape:", getattr(sum_out, "shape", None))
    sum_out[0] = a[:LEN_1D].sum(dtype=np.float64)
    return None

tsvc_2_s311 = s311
