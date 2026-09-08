import numpy as np, time, os
import numba
import numba.np.ufunc.parallel as _npuf

_npuf.set_num_threads(len(os.sched_getaffinity(0)))

@numba.njit()
def _s233_small(aa, bb, cc, n):
    for i in range(8, n):
        for j in range(8, n):
            aa[j, i] = aa[j - 1, i] + cc[j, i]
        for j in range(8, n):
            bb[j, i] = bb[j, i - 1] + cc[j, i]

@numba.njit(parallel=True)
def _aa_strip(aa, cc, n, m, BI):
    NC = (m + BI - 1) // BI
    for c in numba.prange(NC):
        i0 = 8 + c * BI
        i1 = i0 + BI
        if i1 > n: i1 = n
        run = np.empty(i1 - i0, np.float64)
        run[...] = aa[7, i0:i1]
        for j in range(8, n):
            run += cc[j, i0:i1]
            aa[j, i0:i1] = run

@numba.njit(parallel=True)
def _aa_2strip(aa, cc, n, m, BI):
    # each thread interleaves two adjacent strips
    NP = (m + BI - 1) // (2 * BI)
    for c in numba.prange(NP):
        i0 = 8 + 2 * c * BI
        im = i0 + BI
        i1 = im + BI
        if i1 > n: i1 = n
        if im > n: im = n
        w0 = im - i0
        w1 = i1 - im
        r0 = np.empty(w0, np.float64); r1 = np.empty(w1, np.float64)
        r0[...] = aa[7, i0:im]
        r1[...] = aa[7, im:i1]
        for j in range(8, n):
            r0 += cc[j, i0:im]
            aa[j, i0:im] = r0
            r1 += cc[j, im:i1]
            aa[j, im:i1] = r1

@numba.njit(parallel=True)
def _aa2a(BRS, cc, n, m, BH):
    NB = (m + BH - 1) // BH
    for k in numba.prange(NB):
        j0 = 8 + k * BH
        j1 = j0 + BH
        if j1 > n: j1 = n
        st = np.empty(m, np.float64)
        st[...] = 0.0
        for j in range(j0, j1):
            for i in range(8, n):
                st[i - 8] += cc[j, i]
        BRS[k, ...] = st

@numba.njit(parallel=True)
def _aa2b(Cpre, BRS, n, m, BH):
    NB = (m + BH - 1) // BH
    for i in numba.prange(8, n):
        run = 0.0
        for k in range(NB):
            Cpre[k, i] = run
            run += BRS[k, i - 8]

@numba.njit(parallel=True)
def _aa2c(aa, cc, Cpre, n, m, BH):
    NB = (m + BH - 1) // BH
    for k in numba.prange(NB):
        j0 = 8 + k * BH
        j1 = j0 + BH
        if j1 > n: j1 = n
        st = np.empty(m, np.float64)
        base = np.empty(m, np.float64)
        st[...] = 0.0
        for i in range(8, n):
            base[i - 8] = aa[7, i] + Cpre[k, i]
        for j in range(j0, j1):
            for i in range(8, n):
                s = st[i - 8] + cc[j, i]
                st[i - 8] = s
                aa[j, i] = base[i - 8] + s

def _aa_2phase(aa, cc, n, m, BH):
    BRS = np.zeros(((m + BH - 1) // BH, m))
    Cpre = np.zeros(((m + BH - 1) // BH + 1, n))
    _aa2a(BRS, cc, n, m, BH)
    _aa2b(Cpre, BRS, n, m, BH)
    _aa2c(aa, cc, Cpre, n, m, BH)

@numba.njit(parallel=True)
def _bb_pass(bb, cc, n):
    for j in numba.prange(8, n):
        run = bb[j, 7]
        for i in range(8, n):
            run += cc[j, i]
            bb[j, i] = run

def t(f, reps=3):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); f(); t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best

probed = {'v': False}

def s233(aa, bb, cc, LEN_2D):
    n = int(LEN_2D)
    if n <= 8:
        return None
    m = n - 8
    if n <= 256:
        _s233_small(aa, bb, cc, n)
        if not probed['v']:
            probed['v'] = True
            d64 = np.ones((64, 64))
            print(f"SMALL n={n}: {t(lambda: _s233_small(d64, d64.copy(), d64.copy(), 64), 9)*1e6:.2f} us", flush=True)
        return None
    if n == 7169 and not probed['v']:
        probed['v'] = True
        aa0 = aa.copy()
        print("probe start", flush=True)
        for BH in (32, 64, 128):
            print(f"2phase BH={BH}: {t(lambda: _aa_2phase(aa, cc, n, m, BH))*1e3:.2f} ms", flush=True)
        print(f"strip256: {t(lambda: _aa_strip(aa, cc, n, m, 256))*1e3:.2f} ms", flush=True)
        print(f"strip512: {t(lambda: _aa_strip(aa, cc, n, m, 512))*1e3:.2f} ms", flush=True)
        print(f"2strip128: {t(lambda: _aa_2strip(aa, cc, n, m, 128))*1e3:.2f} ms", flush=True)
        print(f"bb: {t(lambda: _bb_pass(bb, cc, n))*1e3:.2f} ms", flush=True)
        print(f"small64: {t(lambda: _s233_small(aa0[:64,:64], aa0[:64,:64].copy(), aa0[:64,:64].copy(), 64), 9)*1e6:.2f} us", flush=True)
        del aa0
    _aa_2phase(aa, cc, n, m, 64)
    _bb_pass(bb, cc, n)
    return None

tsvc_2_s233 = s233

def _warm():
    d = np.ones((16, 16))
    _s233_small(d, d.copy(), d.copy(), 16)
    _aa_strip(d, d.copy(), 16, 8, 32)
    _aa_2strip(d, d.copy(), 16, 8, 32)
    _aa2a(np.zeros((1, 8)), d.copy(), 16, 8, 8)
    _aa2b(np.zeros((2, 16)), np.zeros((1, 8)), 16, 8, 8)
    _aa2c(d, d.copy(), np.zeros((2, 16)), 16, 8, 8)
    _bb_pass(d, d.copy(), 16)

_warm()
