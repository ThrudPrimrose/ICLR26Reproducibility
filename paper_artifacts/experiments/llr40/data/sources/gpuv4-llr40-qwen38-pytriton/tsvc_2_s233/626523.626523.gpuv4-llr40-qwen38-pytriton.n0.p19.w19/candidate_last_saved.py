import sys, os, time
import numpy as np
import numba
from concurrent.futures import ThreadPoolExecutor

@numba.njit(parallel=True)
def aa_coltile(aa, cc, N, T):
    nt = (N - 8 + T - 1) // T
    for t in numba.prange(nt):
        i0 = 8 + t * T
        n = min(T, N - i0)
        acc = np.empty(n)
        for c in range(n):
            acc[c] = aa[7, i0 + c]
        for j in range(8, N):
            for c in range(n):
                acc[c] += cc[j, i0 + c]
                aa[j, i0 + c] = acc[c]
@numba.njit(parallel=True)
def bb_row(bb, cc, N):
    for j in numba.prange(8, N):
        acc = bb[j, 7]
        for i in range(8, N):
            acc += cc[j, i]
            bb[j, i] = acc

def s233(aa, bb, cc, LEN_2D):
    N = int(LEN_2D)
    out = []
    def best(f, nt, reps=4):
        numba.set_num_threads(nt)
        f()
        ts=[]
        for _ in range(reps):
            t0=time.perf_counter(); f(); ts.append(time.perf_counter()-t0)
        return min(ts)
    T=512
    a1=np.zeros_like(aa); b1=np.zeros_like(bb)
    t_aa = best(lambda: aa_coltile(a1, cc, N, T), 24)
    t_bb = best(lambda: bb_row(b1, cc, N), 24)
    # sequential
    def seq():
        numba.set_num_threads(24); aa_coltile(a1, cc, N, T); bb_row(b1, cc, N)
    t_seq = best(seq, 24)
    # parallel via python threads
    ex = ThreadPoolExecutor(max_workers=2)
    def par():
        numba.set_num_threads(12)
        f1 = ex.submit(aa_coltile, a1, cc, N, T)
        f2 = ex.submit(bb_row, b1, cc, N)
        f1.result(); f2.result()
    t_par = best(par, 12)
    out.append(f"T512: aa={t_aa*1e3:.1f}ms bb={t_bb*1e3:.1f}ms")
    out.append(f"sequential={t_seq*1e3:.1f}ms  parallel(12+12)={t_par*1e3:.1f}ms")
    print("\n".join(out), file=sys.stdout); sys.stdout.flush()
    return None
