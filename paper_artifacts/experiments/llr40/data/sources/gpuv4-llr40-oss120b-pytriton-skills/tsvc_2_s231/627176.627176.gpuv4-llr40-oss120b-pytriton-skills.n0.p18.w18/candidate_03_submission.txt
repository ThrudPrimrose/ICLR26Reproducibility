import numpy as np
import numba

# Serial Numba implementation (no parallel)
@numba.njit(fastmath=True)
def _s231_serial(aa, bb, LEN_2D):
    for i in range(LEN_2D):
        for j in range(1, LEN_2D):
            aa[j, i] = aa[j - 1, i] + bb[j, i]

# Parallel Numba implementation (parallel across columns)
@numba.njit(parallel=True, fastmath=True)
def _s231_parallel(aa, bb, LEN_2D):
    for i in numba.prange(LEN_2D):
        for j in range(1, LEN_2D):
            aa[j, i] = aa[j - 1, i] + bb[j, i]

# Pre‑compile both versions for common dtypes to avoid JIT overhead during timed call.
# Compile for int64 (typical integer data) and float32 (typical floating data).
_dummy_int = np.empty((1, 1), dtype=np.int64)
_dummy_float = np.empty((1, 1), dtype=np.float32)
# Serial compile
_s231_serial(_dummy_int, _dummy_int, 1)
_s231_serial(_dummy_float, _dummy_float, 1)
# Parallel compile
_s231_parallel(_dummy_int, _dummy_int, 1)
_s231_parallel(_dummy_float, _dummy_float, 1)

def s231(aa, bb, LEN_2D):
    """In‑place prefix sum kernel.
    Uses a parallel Numba implementation when more than one thread is available;
    otherwise falls back to a serial approach (Numba or NumPy) to avoid parallel overhead.
    """
    # Fast path for trivial sizes.
    if LEN_2D <= 1:
        return None
    # Choose implementation based on the number of threads Numba will use.
    # ``numba.get_num_threads`` reports the thread count for parallel regions.
    if numba.get_num_threads() > 1:
        _s231_parallel(aa, bb, LEN_2D)
    else:
        # Serial path – use the compiled Numba serial implementation.
        # (no extra compilation needed)
        _s231_serial(aa, bb, LEN_2D)
        # No additional operation needed (serial version handled all updates)
    return None
