"""TSVC tsvc_2 kernel s2710 -- optimized Python (numba) implementation.

In-place ABI: mutates a, b, c like the reference; returns None.
All four semantic variants are precompiled at import time so the timed
region contains no JIT compilation.
"""
import numpy as np
from numba import njit, prange

_J = dict(nogil=True, boundscheck=False)


# ---- n > 10, x[0] > 0 ----
@njit(parallel=True, **_J)
def _p_big_pos(a, b, c, d, e, n):
    for i in prange(n):
        ai = a[i]; bi = b[i]; di = d[i]; ei = e[i]
        if ai > bi:
            a[i] = ai + bi * di
            c[i] = c[i] + di * di
        else:
            b[i] = ai + ei * ei
            c[i] = ai + di * di


# ---- n > 10, x[0] <= 0 ----
@njit(parallel=True, **_J)
def _p_big_neg(a, b, c, d, e, n):
    for i in prange(n):
        ai = a[i]; bi = b[i]; di = d[i]; ei = e[i]
        if ai > bi:
            a[i] = ai + bi * di
            c[i] = c[i] + di * di
        else:
            b[i] = ai + ei * ei
            c[i] = c[i] + ei * ei


# ---- n <= 10, x[0] > 0 ----
@njit(parallel=True, **_J)
def _p_small_pos(a, b, c, d, e, n):
    for i in prange(n):
        ai = a[i]; bi = b[i]; di = d[i]; ei = e[i]
        if ai > bi:
            a[i] = ai + bi * di
            c[i] = di * ei + 1.0
        else:
            b[i] = ai + ei * ei
            c[i] = ai + di * di


# ---- n <= 10, x[0] <= 0 ----
@njit(parallel=True, **_J)
def _p_small_neg(a, b, c, d, e, n):
    for i in prange(n):
        ai = a[i]; bi = b[i]; ci = c[i]; di = d[i]; ei = e[i]
        if ai > bi:
            a[i] = ai + bi * di
            c[i] = di * ei + 1.0
        else:
            b[i] = ai + ei * ei
            c[i] = ci + ei * ei


# ---- scalar (single-thread) variants for small n ----
@njit(**_J)
def _s_big_pos(a, b, c, d, e, n):
    for i in range(n):
        ai = a[i]; bi = b[i]; di = d[i]; ei = e[i]
        if ai > bi:
            a[i] = ai + bi * di
            c[i] = c[i] + di * di
        else:
            b[i] = ai + ei * ei
            c[i] = ai + di * di


@njit(**_J)
def _s_big_neg(a, b, c, d, e, n):
    for i in range(n):
        ai = a[i]; bi = b[i]; di = d[i]; ei = e[i]
        if ai > bi:
            a[i] = ai + bi * di
            c[i] = c[i] + di * di
        else:
            b[i] = ai + ei * ei
            c[i] = c[i] + ei * ei


@njit(**_J)
def _s_small_pos(a, b, c, d, e, n):
    for i in range(n):
        ai = a[i]; bi = b[i]; di = d[i]; ei = e[i]
        if ai > bi:
            a[i] = ai + bi * di
            c[i] = di * ei + 1.0
        else:
            b[i] = ai + ei * ei
            c[i] = ai + di * di


@njit(**_J)
def _s_small_neg(a, b, c, d, e, n):
    for i in range(n):
        ai = a[i]; bi = b[i]; ci = c[i]; di = d[i]; ei = e[i]
        if ai > bi:
            a[i] = ai + bi * di
            c[i] = di * ei + 1.0
        else:
            b[i] = ai + ei * ei
            c[i] = ci + ei * ei


_PAR = {True: {True: _p_big_pos, False: _p_big_neg},
        False: {True: _p_small_pos, False: _p_small_neg}}
_SCAL = {True: {True: _s_big_pos, False: _s_big_neg},
         False: {True: _s_small_pos, False: _s_small_neg}}

_SMALL_N = 1 << 15  # below this the OMP round-trip is not worth it


def s2710(a, b, c, d, e, x, LEN_1D):
    n = int(LEN_1D)
    if n <= 0:
        return None
    x0 = float(x[0]) if x.size else 0.0
    tab = _SCAL if n < _SMALL_N else _PAR
    tab[n > 10][x0 > 0.0](a, b, c, d, e, n)


# ---------------- import-time warmup (not timed) ----------------
def _warmup():
    rng = np.random.default_rng(0)
    for nn in (4, 10, 1024, _SMALL_N + 64):
        for x0 in (1.0, -1.0):
            a = rng.standard_normal(nn); b = rng.standard_normal(nn)
            c = rng.standard_normal(nn); d = rng.standard_normal(nn)
            e = rng.standard_normal(nn); x = np.array([x0])
            s2710(a, b, c, d, e, x, nn)


_warmup()
