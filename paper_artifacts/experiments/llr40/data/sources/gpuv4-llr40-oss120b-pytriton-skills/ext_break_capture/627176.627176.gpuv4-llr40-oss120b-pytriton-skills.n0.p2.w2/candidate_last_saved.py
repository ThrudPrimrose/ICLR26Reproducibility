import numpy as np

def ext_break_capture(a, out_index, out_value, LEN_1D, K):
    """Find first index where a[i] > K and capture value.
    In-place writes into ``out_index`` and ``out_value``.
    Implements a block-wise scan to stop early, reducing work
    compared to a full-array scan.
    """
    # Sentinel values (as per reference)
    out_index[0] = -1
    if np.issubdtype(out_value.dtype, np.floating):
        out_value[0] = -1.0
    else:
        out_value[0] = out_value.dtype.type(-1)

    # Choose a block size that balances Python overhead and cache usage.
    # 1 MiB of float64 elements (~8 MiB) is a reasonable default. Use 1<<20.
    BLOCK = 1 << 20  # 1,048,576 elements
    for start in range(0, LEN_1D, BLOCK):
        end = start + BLOCK
        if end > LEN_1D:
            end = LEN_1D
        sub = a[start:end]
        # If no element in this block exceeds K, skip.
        if sub.max() <= K:
            continue
        # This block contains a value > K. Find the first such index.
        idx_local = np.argmax(sub > K)
        if sub[idx_local] > K:
            out_index[0] = start + idx_local
            out_value[0] = sub[idx_local]
        break
    return None
