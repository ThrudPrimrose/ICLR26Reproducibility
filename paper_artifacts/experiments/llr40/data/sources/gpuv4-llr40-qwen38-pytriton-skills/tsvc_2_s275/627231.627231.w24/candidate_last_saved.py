"""TSVC tsvc_2 s275 -- column scan with block interleaving, numba parallel."""
import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=False, boundscheck=False)
def _s275_core(aa, bb, cc, N):
    nchunk = N // 8
    nfull = nchunk // 4
    for t in prange(nfull):
        i0 = t * 32

        act0 = aa[0, i0+0] > 0.0; s0 = aa[0, i0+0]
        act1 = aa[0, i0+1] > 0.0; s1 = aa[0, i0+1]
        act2 = aa[0, i0+2] > 0.0; s2 = aa[0, i0+2]
        act3 = aa[0, i0+3] > 0.0; s3 = aa[0, i0+3]
        act4 = aa[0, i0+4] > 0.0; s4 = aa[0, i0+4]
        act5 = aa[0, i0+5] > 0.0; s5 = aa[0, i0+5]
        act6 = aa[0, i0+6] > 0.0; s6 = aa[0, i0+6]
        act7 = aa[0, i0+7] > 0.0; s7 = aa[0, i0+7]
        act8 = aa[0, i0+8] > 0.0; s8 = aa[0, i0+8]
        act9 = aa[0, i0+9] > 0.0; s9 = aa[0, i0+9]
        act10 = aa[0, i0+10] > 0.0; s10 = aa[0, i0+10]
        act11 = aa[0, i0+11] > 0.0; s11 = aa[0, i0+11]
        act12 = aa[0, i0+12] > 0.0; s12 = aa[0, i0+12]
        act13 = aa[0, i0+13] > 0.0; s13 = aa[0, i0+13]
        act14 = aa[0, i0+14] > 0.0; s14 = aa[0, i0+14]
        act15 = aa[0, i0+15] > 0.0; s15 = aa[0, i0+15]
        act16 = aa[0, i0+16] > 0.0; s16 = aa[0, i0+16]
        act17 = aa[0, i0+17] > 0.0; s17 = aa[0, i0+17]
        act18 = aa[0, i0+18] > 0.0; s18 = aa[0, i0+18]
        act19 = aa[0, i0+19] > 0.0; s19 = aa[0, i0+19]
        act20 = aa[0, i0+20] > 0.0; s20 = aa[0, i0+20]
        act21 = aa[0, i0+21] > 0.0; s21 = aa[0, i0+21]
        act22 = aa[0, i0+22] > 0.0; s22 = aa[0, i0+22]
        act23 = aa[0, i0+23] > 0.0; s23 = aa[0, i0+23]
        act24 = aa[0, i0+24] > 0.0; s24 = aa[0, i0+24]
        act25 = aa[0, i0+25] > 0.0; s25 = aa[0, i0+25]
        act26 = aa[0, i0+26] > 0.0; s26 = aa[0, i0+26]
        act27 = aa[0, i0+27] > 0.0; s27 = aa[0, i0+27]
        act28 = aa[0, i0+28] > 0.0; s28 = aa[0, i0+28]
        act29 = aa[0, i0+29] > 0.0; s29 = aa[0, i0+29]
        act30 = aa[0, i0+30] > 0.0; s30 = aa[0, i0+30]
        act31 = aa[0, i0+31] > 0.0; s31 = aa[0, i0+31]
        for j in range(1, N):
            if act0:
                s0 += bb[j, i0+0] * cc[j, i0+0]; aa[j, i0+0] = s0
            if act1:
                s1 += bb[j, i0+1] * cc[j, i0+1]; aa[j, i0+1] = s1
            if act2:
                s2 += bb[j, i0+2] * cc[j, i0+2]; aa[j, i0+2] = s2
            if act3:
                s3 += bb[j, i0+3] * cc[j, i0+3]; aa[j, i0+3] = s3
            if act4:
                s4 += bb[j, i0+4] * cc[j, i0+4]; aa[j, i0+4] = s4
            if act5:
                s5 += bb[j, i0+5] * cc[j, i0+5]; aa[j, i0+5] = s5
            if act6:
                s6 += bb[j, i0+6] * cc[j, i0+6]; aa[j, i0+6] = s6
            if act7:
                s7 += bb[j, i0+7] * cc[j, i0+7]; aa[j, i0+7] = s7
            if act8:
                s8 += bb[j, i0+8] * cc[j, i0+8]; aa[j, i0+8] = s8
            if act9:
                s9 += bb[j, i0+9] * cc[j, i0+9]; aa[j, i0+9] = s9
            if act10:
                s10 += bb[j, i0+10] * cc[j, i0+10]; aa[j, i0+10] = s10
            if act11:
                s11 += bb[j, i0+11] * cc[j, i0+11]; aa[j, i0+11] = s11
            if act12:
                s12 += bb[j, i0+12] * cc[j, i0+12]; aa[j, i0+12] = s12
            if act13:
                s13 += bb[j, i0+13] * cc[j, i0+13]; aa[j, i0+13] = s13
            if act14:
                s14 += bb[j, i0+14] * cc[j, i0+14]; aa[j, i0+14] = s14
            if act15:
                s15 += bb[j, i0+15] * cc[j, i0+15]; aa[j, i0+15] = s15
            if act16:
                s16 += bb[j, i0+16] * cc[j, i0+16]; aa[j, i0+16] = s16
            if act17:
                s17 += bb[j, i0+17] * cc[j, i0+17]; aa[j, i0+17] = s17
            if act18:
                s18 += bb[j, i0+18] * cc[j, i0+18]; aa[j, i0+18] = s18
            if act19:
                s19 += bb[j, i0+19] * cc[j, i0+19]; aa[j, i0+19] = s19
            if act20:
                s20 += bb[j, i0+20] * cc[j, i0+20]; aa[j, i0+20] = s20
            if act21:
                s21 += bb[j, i0+21] * cc[j, i0+21]; aa[j, i0+21] = s21
            if act22:
                s22 += bb[j, i0+22] * cc[j, i0+22]; aa[j, i0+22] = s22
            if act23:
                s23 += bb[j, i0+23] * cc[j, i0+23]; aa[j, i0+23] = s23
            if act24:
                s24 += bb[j, i0+24] * cc[j, i0+24]; aa[j, i0+24] = s24
            if act25:
                s25 += bb[j, i0+25] * cc[j, i0+25]; aa[j, i0+25] = s25
            if act26:
                s26 += bb[j, i0+26] * cc[j, i0+26]; aa[j, i0+26] = s26
            if act27:
                s27 += bb[j, i0+27] * cc[j, i0+27]; aa[j, i0+27] = s27
            if act28:
                s28 += bb[j, i0+28] * cc[j, i0+28]; aa[j, i0+28] = s28
            if act29:
                s29 += bb[j, i0+29] * cc[j, i0+29]; aa[j, i0+29] = s29
            if act30:
                s30 += bb[j, i0+30] * cc[j, i0+30]; aa[j, i0+30] = s30
            if act31:
                s31 += bb[j, i0+31] * cc[j, i0+31]; aa[j, i0+31] = s31
    for i in prange(nfull * 32, N):
        if aa[0, i] > 0.0:
            s = aa[0, i]
            for j in range(1, N):
                s += bb[j, i] * cc[j, i]
                aa[j, i] = s


def s275(aa, bb, cc, LEN_2D):
    _s275_core(aa, bb, cc, int(LEN_2D))


# ---- import-time warmup: compile all variants before the timer starts ----
try:
    _w = np.ones((64, 64))
    _s275_core(_w, _w.copy(), _w.copy(), 64)
    _w32 = np.ones((64, 64), dtype=np.float32)
    _s275_core(_w32, _w32.copy(), _w32.copy(), 64)
except Exception:
    pass
