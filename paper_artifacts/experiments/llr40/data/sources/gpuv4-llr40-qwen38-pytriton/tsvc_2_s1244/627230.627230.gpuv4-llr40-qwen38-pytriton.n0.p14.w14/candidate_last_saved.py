import numpy as np

def s1244(a, b, c, d, LEN_1D):
    a_new = b[:-1] + c[:-1] * c[:-1] + b[:-1] * b[:-1] + c[:-1]
    d[:-1] = a_new + a[1:]
    a[:-1] = a_new
    return None
