import numba
import numpy as np

@numba.njit(fastmath=True)
def _wf_diff_skew(a, LEN_2D):
    for i in range(1, LEN_2D):
        for j in range(LEN_2D - 1):
            a[i, j] = a[i, j] + a[i - 1, j] + a[i - 1, j + 1]
    return

def wf_diff_skew(a, LEN_2D):
    """In-place wavefront diff using a fastmath Numba JIT implementation.
+
+    Args:
+        a (np.ndarray): 2‑D array of shape ``(LEN_2D, LEN_2D)``. Modified in place.
+        LEN_2D (int): Length of each dimension.
+    """
    _wf_diff_skew(a, LEN_2D)
    return None
