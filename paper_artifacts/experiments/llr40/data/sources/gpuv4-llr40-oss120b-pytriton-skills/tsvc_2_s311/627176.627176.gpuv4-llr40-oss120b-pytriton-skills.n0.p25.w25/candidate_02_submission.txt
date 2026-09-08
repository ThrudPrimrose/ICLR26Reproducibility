import numpy as np

def s311(a, sum_out, LEN_1D):
    """Sum the elements of a into sum_out[0]. In-place version.
    Parameters
    ----------
    a : np.ndarray
        Input 1D array of length LEN_1D.
    sum_out : np.ndarray
        Output array where the scalar sum is stored at index 0.
    LEN_1D : int
        Number of elements of a to sum.
    """
    sum_out[0] = np.sum(a[:LEN_1D])
