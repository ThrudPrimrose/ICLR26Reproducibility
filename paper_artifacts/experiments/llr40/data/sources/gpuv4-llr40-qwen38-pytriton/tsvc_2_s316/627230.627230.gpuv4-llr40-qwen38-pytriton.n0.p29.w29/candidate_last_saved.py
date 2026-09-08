"""Optimized tsvc_2 s316: single-pass min reduction with the reference's
exact comparison semantics.

Reference (C):
    x = a[0]
    for i in 1..LEN_1D-1:
        if a[i] < x:
            x = a[i]
    result[0] = x

The C loop never adopts a NaN (comparison with NaN is false), and keeps the
previously held value on ties.  np.fmin ignores NaN in exactly the same way
(the two-arg fmin returns the non-NaN operand), and the final `m < x` guard
reproduces the tie-keeps-old behavior, so the result matches the C reference
for finite values, +/-inf, +/-0 (up to the sign of zero, which is
comparison-equal) and NaN inputs.
"""
import numpy as np


def s316(a, result, LEN_1D):
    x = a[0]
    if LEN_1D > 1:
        m = np.fmin.reduce(a[1:LEN_1D])
        if m < x:
            x = m
    result[0] = x
