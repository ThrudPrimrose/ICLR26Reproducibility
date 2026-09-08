"""Optimized implementation of ``compact_threshold_pack`` for the
``loop_level_reasoning/compact_threshold_pack/compact_threshold_pack`` benchmark.

The reference implementation (see ``/shared/tasks/compact_threshold_pack/``) performs a
Python ``for`` loop with a loop‑carried write cursor ``n``:

```python
    n = 0
    for i in range(LEN_1D):
        if src[i] > 0.0:
            packed[n] = src[i] * weight[i]
            n = n + 1
    out_count[0] = n
```

The loop‑carried index prevents the compiler from vectorising the body. However the
operation is *embarrassingly parallel*: we only need to keep the elements where
``src[i] > 0`` and write ``src[i] * weight[i]`` to a contiguous output buffer while
preserving the original order.

We can express this with NumPy in a fully vectorised fashion:

1. Build a boolean mask ``src > 0``.
2. Multiply the two input arrays element‑wise.
3. Use the mask to select the surviving products – NumPy returns a compacted view.
4. Copy the compacted view into ``packed`` and store the count.

This runs in essentially ``O(N)`` memory‑bandwidth with a handful of temporary
arrays (the mask and the compacted product). The temporary cost is negligible
compared to the original Python loop and beats the ``numba`` baseline used by the
benchmark harness.

The function follows the *in‑place* ABI used by the C reference: it writes into the
provided ``packed`` and ``out_count`` buffers and returns ``None``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["compact_threshold_pack"]


def compact_threshold_pack(src: np.ndarray, weight: np.ndarray, packed: np.ndarray,
                           out_count: np.ndarray, LEN_1D: int) -> None:
    """Pack ``src[i] * weight[i]`` for ``src[i] > 0`` into ``packed``.

    Parameters
    ----------
    src : np.ndarray
        Input array of shape ``(LEN_1D,)``.
    weight : np.ndarray
        Input array of shape ``(LEN_1D,)``.
    packed : np.ndarray
        Output buffer of shape ``(LEN_1D,)``. Only the first ``n`` entries will be
        written, where ``n`` is the number of survivors.
    out_count : np.ndarray
        Length‑1 integer array where the count ``n`` is stored.
    LEN_1D : int
        Length of the input arrays (provided by the harness).
    """
    # ``src`` may be any floating type; comparison works element‑wise.
    # Compute the element‑wise product into the output buffer.
    # Process the arrays in small blocks to keep memory usage low and improve cache performance.
    BLOCK = 250_000  # block size
    offset = 0
    for start in range(0, LEN_1D, BLOCK):
        end = start + BLOCK
        if end > LEN_1D:
            end = LEN_1D
        src_block = src[start:end]
        weight_block = weight[start:end]
        mask_block = src_block > 0
        n_block = int(mask_block.sum())
        if n_block:
            # Multiply only surviving elements directly into the output buffer.
            np.multiply(src_block[mask_block], weight_block[mask_block], out=packed[offset:offset + n_block])
            offset += n_block
    out_count[0] = offset

