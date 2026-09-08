import os
import numpy as np
import numba as nb

@nb.njit(parallel=True)
def _k_x0pos(a, b, c, d, e, n):
    n8 = n - (n % 8)
    for i in nb.prange(n8 // 8):
        j = i * 8
        a_0 = a[j+0]; b_0 = b[j+0]; c_0 = c[j+0]; d_0 = d[j+0]; e_0 = e[j+0]
        a_1 = a[j+1]; b_1 = b[j+1]; c_1 = c[j+1]; d_1 = d[j+1]; e_1 = e[j+1]
        a_2 = a[j+2]; b_2 = b[j+2]; c_2 = c[j+2]; d_2 = d[j+2]; e_2 = e[j+2]
        a_3 = a[j+3]; b_3 = b[j+3]; c_3 = c[j+3]; d_3 = d[j+3]; e_3 = e[j+3]
        a_4 = a[j+4]; b_4 = b[j+4]; c_4 = c[j+4]; d_4 = d[j+4]; e_4 = e[j+4]
        a_5 = a[j+5]; b_5 = b[j+5]; c_5 = c[j+5]; d_5 = d[j+5]; e_5 = e[j+5]
        a_6 = a[j+6]; b_6 = b[j+6]; c_6 = c[j+6]; d_6 = d[j+6]; e_6 = e[j+6]
        a_7 = a[j+7]; b_7 = b[j+7]; c_7 = c[j+7]; d_7 = d[j+7]; e_7 = e[j+7]
        m_0 = 1.0 if a_0 > b_0 else 0.0
        nm_0 = 1.0 - m_0
        m_1 = 1.0 if a_1 > b_1 else 0.0
        nm_1 = 1.0 - m_1
        m_2 = 1.0 if a_2 > b_2 else 0.0
        nm_2 = 1.0 - m_2
        m_3 = 1.0 if a_3 > b_3 else 0.0
        nm_3 = 1.0 - m_3
        m_4 = 1.0 if a_4 > b_4 else 0.0
        nm_4 = 1.0 - m_4
        m_5 = 1.0 if a_5 > b_5 else 0.0
        nm_5 = 1.0 - m_5
        m_6 = 1.0 if a_6 > b_6 else 0.0
        nm_6 = 1.0 - m_6
        m_7 = 1.0 if a_7 > b_7 else 0.0
        nm_7 = 1.0 - m_7
        a[j+0] = a_0 + m_0 * (b_0 * d_0)
        b[j+0] = m_0 * b_0 + nm_0 * (a_0 + e_0 * e_0)
        c[j+0] = m_0 * c_0 + nm_0 * a_0 + d_0 * d_0
        a[j+1] = a_1 + m_1 * (b_1 * d_1)
        b[j+1] = m_1 * b_1 + nm_1 * (a_1 + e_1 * e_1)
        c[j+1] = m_1 * c_1 + nm_1 * a_1 + d_1 * d_1
        a[j+2] = a_2 + m_2 * (b_2 * d_2)
        b[j+2] = m_2 * b_2 + nm_2 * (a_2 + e_2 * e_2)
        c[j+2] = m_2 * c_2 + nm_2 * a_2 + d_2 * d_2
        a[j+3] = a_3 + m_3 * (b_3 * d_3)
        b[j+3] = m_3 * b_3 + nm_3 * (a_3 + e_3 * e_3)
        c[j+3] = m_3 * c_3 + nm_3 * a_3 + d_3 * d_3
        a[j+4] = a_4 + m_4 * (b_4 * d_4)
        b[j+4] = m_4 * b_4 + nm_4 * (a_4 + e_4 * e_4)
        c[j+4] = m_4 * c_4 + nm_4 * a_4 + d_4 * d_4
        a[j+5] = a_5 + m_5 * (b_5 * d_5)
        b[j+5] = m_5 * b_5 + nm_5 * (a_5 + e_5 * e_5)
        c[j+5] = m_5 * c_5 + nm_5 * a_5 + d_5 * d_5
        a[j+6] = a_6 + m_6 * (b_6 * d_6)
        b[j+6] = m_6 * b_6 + nm_6 * (a_6 + e_6 * e_6)
        c[j+6] = m_6 * c_6 + nm_6 * a_6 + d_6 * d_6
        a[j+7] = a_7 + m_7 * (b_7 * d_7)
        b[j+7] = m_7 * b_7 + nm_7 * (a_7 + e_7 * e_7)
        c[j+7] = m_7 * c_7 + nm_7 * a_7 + d_7 * d_7
    for j in range(n8, n):
        aj = a[j]; bj = b[j]; cj = c[j]; dj = d[j]; ej = e[j]
        m = 1.0 if aj > bj else 0.0
        nm = 1.0 - m
        a[j] = aj + m * (bj * dj)
        b[j] = m * bj + nm * (aj + ej * ej)
        c[j] = m * cj + nm * aj + dj * dj

@nb.njit(parallel=True)
def _k_x0neg(a, b, c, d, e, n):
    n8 = n - (n % 8)
    for i in nb.prange(n8 // 8):
        j = i * 8
        a_0 = a[j+0]; b_0 = b[j+0]; c_0 = c[j+0]; d_0 = d[j+0]; e_0 = e[j+0]
        a_1 = a[j+1]; b_1 = b[j+1]; c_1 = c[j+1]; d_1 = d[j+1]; e_1 = e[j+1]
        a_2 = a[j+2]; b_2 = b[j+2]; c_2 = c[j+2]; d_2 = d[j+2]; e_2 = e[j+2]
        a_3 = a[j+3]; b_3 = b[j+3]; c_3 = c[j+3]; d_3 = d[j+3]; e_3 = e[j+3]
        a_4 = a[j+4]; b_4 = b[j+4]; c_4 = c[j+4]; d_4 = d[j+4]; e_4 = e[j+4]
        a_5 = a[j+5]; b_5 = b[j+5]; c_5 = c[j+5]; d_5 = d[j+5]; e_5 = e[j+5]
        a_6 = a[j+6]; b_6 = b[j+6]; c_6 = c[j+6]; d_6 = d[j+6]; e_6 = e[j+6]
        a_7 = a[j+7]; b_7 = b[j+7]; c_7 = c[j+7]; d_7 = d[j+7]; e_7 = e[j+7]
        m_0 = 1.0 if a_0 > b_0 else 0.0
        nm_0 = 1.0 - m_0
        m_1 = 1.0 if a_1 > b_1 else 0.0
        nm_1 = 1.0 - m_1
        m_2 = 1.0 if a_2 > b_2 else 0.0
        nm_2 = 1.0 - m_2
        m_3 = 1.0 if a_3 > b_3 else 0.0
        nm_3 = 1.0 - m_3
        m_4 = 1.0 if a_4 > b_4 else 0.0
        nm_4 = 1.0 - m_4
        m_5 = 1.0 if a_5 > b_5 else 0.0
        nm_5 = 1.0 - m_5
        m_6 = 1.0 if a_6 > b_6 else 0.0
        nm_6 = 1.0 - m_6
        m_7 = 1.0 if a_7 > b_7 else 0.0
        nm_7 = 1.0 - m_7
        a[j+0] = a_0 + m_0 * (b_0 * d_0)
        b[j+0] = m_0 * b_0 + nm_0 * (a_0 + e_0 * e_0)
        c[j+0] = c_0 + m_0 * (d_0 * d_0) + nm_0 * (e_0 * e_0)
        a[j+1] = a_1 + m_1 * (b_1 * d_1)
        b[j+1] = m_1 * b_1 + nm_1 * (a_1 + e_1 * e_1)
        c[j+1] = c_1 + m_1 * (d_1 * d_1) + nm_1 * (e_1 * e_1)
        a[j+2] = a_2 + m_2 * (b_2 * d_2)
        b[j+2] = m_2 * b_2 + nm_2 * (a_2 + e_2 * e_2)
        c[j+2] = c_2 + m_2 * (d_2 * d_2) + nm_2 * (e_2 * e_2)
        a[j+3] = a_3 + m_3 * (b_3 * d_3)
        b[j+3] = m_3 * b_3 + nm_3 * (a_3 + e_3 * e_3)
        c[j+3] = c_3 + m_3 * (d_3 * d_3) + nm_3 * (e_3 * e_3)
        a[j+4] = a_4 + m_4 * (b_4 * d_4)
        b[j+4] = m_4 * b_4 + nm_4 * (a_4 + e_4 * e_4)
        c[j+4] = c_4 + m_4 * (d_4 * d_4) + nm_4 * (e_4 * e_4)
        a[j+5] = a_5 + m_5 * (b_5 * d_5)
        b[j+5] = m_5 * b_5 + nm_5 * (a_5 + e_5 * e_5)
        c[j+5] = c_5 + m_5 * (d_5 * d_5) + nm_5 * (e_5 * e_5)
        a[j+6] = a_6 + m_6 * (b_6 * d_6)
        b[j+6] = m_6 * b_6 + nm_6 * (a_6 + e_6 * e_6)
        c[j+6] = c_6 + m_6 * (d_6 * d_6) + nm_6 * (e_6 * e_6)
        a[j+7] = a_7 + m_7 * (b_7 * d_7)
        b[j+7] = m_7 * b_7 + nm_7 * (a_7 + e_7 * e_7)
        c[j+7] = c_7 + m_7 * (d_7 * d_7) + nm_7 * (e_7 * e_7)
    for j in range(n8, n):
        aj = a[j]; bj = b[j]; cj = c[j]; dj = d[j]; ej = e[j]
        m = 1.0 if aj > bj else 0.0
        nm = 1.0 - m
        a[j] = aj + m * (bj * dj)
        b[j] = m * bj + nm * (aj + ej * ej)
        c[j] = cj + m * (dj * dj) + nm * (ej * ej)

def _small(a, b, c, d, e, x, n):
    # n <= 10: reference semantics exactly (loop-invariant branches hoisted)
    xx = x[0] > 0.0
    for i in range(n):
        if a[i] > b[i]:
            a[i] = a[i] + b[i] * d[i]
            c[i] = d[i] * e[i] + 1.0
        else:
            b[i] = a[i] + e[i] * e[i]
            if xx:
                c[i] = a[i] + d[i] * d[i]
            else:
                c[i] = c[i] + e[i] * e[i]


def _fallback(a, b, c, d, e, x, n):
    a = a[:n]; b = b[:n]; c = c[:n]; d = d[:n]; e = e[:n]
    m = a > b
    nn = ~m
    a_m = a[m]; b_m = b[m]; d_m = d[m]; e_m = e[m]; c_m = c[m]
    a_n = a[nn]; d_n = d[nn]; e_n = e[nn]; c_n = c[nn]
    a[m] = a_m + b_m * d_m
    if n > 10:
        c[m] = c_m + d_m * d_m
    else:
        c[m] = d_m * e_m + 1.0
    b[nn] = a_n + e_n * e_n
    if x[0] > 0.0:
        c[nn] = a_n + d_n * d_n
    else:
        c[nn] = c_n + e_n * e_n


def s2710(a, b, c, d, e, x, LEN_1D):
    n = int(LEN_1D)
    if n <= 0:
        return
    if n <= 10:
        _small(a[:n], b[:n], c[:n], d[:n], e[:n], x, n)
        return
    if x[0] > 0.0:
        _k_x0pos(a[:n], b[:n], c[:n], d[:n], e[:n], n)
    else:
        _k_x0neg(a[:n], b[:n], c[:n], d[:n], e[:n], n)


def _warm():
    try:
        nb.set_num_threads(max(1, len(os.sched_getaffinity(0))))
    except Exception:
        pass
    rng = np.random.default_rng(0)
    N = 1 << 14
    a = rng.random(N); b = rng.random(N); c = rng.random(N)
    d = rng.random(N); e = rng.random(N); x = np.array([1.0])
    for _ in range(2):
        _k_x0pos(a, b, c, d, e, N)
        _k_x0neg(a, b, c, d, e, N)
    a2 = rng.random(7); b2 = rng.random(7); c2 = rng.random(7)
    d2 = rng.random(7); e2 = rng.random(7)
    _small(a2, b2, c2, d2, e2, np.array([-1.0]), 7)
    _fallback(a.copy(), b.copy(), c.copy(), d, e, x, N)


_warm()
