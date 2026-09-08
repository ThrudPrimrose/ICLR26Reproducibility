import numpy as np

def argmax_with_index(a, out_value, out_index, LEN_1D):
    """Find the maximum value and its index in a 1D array.
    Parameters
    ----------
    a : numpy.ndarray
        Input array of shape (LEN_1D,).
    out_value : numpy.ndarray
        Output array of shape (1,) to store the maximum value.
    out_index : numpy.ndarray
        Output array of shape (1,) to store the index of the maximum value.
    LEN_1D : int
        Length of the input array (should match a.shape[0]).
    """
    # Use numpy's efficient max and argmax (both O(N) in C).
    # np.argmax returns the first occurrence of the max, matching the > comparison semantics.
    idx = int(a.argmax())
    out_index[0] = idx
    out_value[0] = a[idx]
    # Return None for in-place semantics.
    return None
