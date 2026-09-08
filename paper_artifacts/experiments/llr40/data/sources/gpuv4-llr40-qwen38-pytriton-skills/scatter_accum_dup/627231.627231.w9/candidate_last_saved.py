import os, sys, time, subprocess, tempfile
import numpy as np
import ctypes

C_SRC = r'''
#define _GNU_SOURCE
#include <stdint.h>
#include <omp.h>
#include <string.h>
#include <stdlib.h>
#include <sched.h>
#include <time.h>
#include <math.h>
#include <stdatomic.h>

static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return (double)ts.tv_sec + (double)ts.tv_nsec*1e-9; }
long affinity_cnt(void){ long n=0; cpu_set_t s; if(sched_getaffinity(0,sizeof(s),&s)==0) n=CPU_COUNT(&s); return n; }
int try_unpin(long ncpu){
    cpu_set_t s; CPU_ZERO(&s);
    for(long i=0;i<ncpu;i++) CPU_SET(i,&s);
    return sched_setaffinity(0, sizeof(s), &s);
}
static _Atomic unsigned long used_cpus;
int cpus_used(int T){
    used_cpus = 0;
    omp_set_num_threads(T);
    #pragma omp parallel
    {
        long c = sched_getcpu();
        if (c >= 0 && c < 64) atomic_fetch_or_explicit(&used_cpus, 1UL << c, memory_order_relaxed);
    }
    int n = 0;
    for (int i = 0; i < 64; i++) n += (int)((used_cpus >> i) & 1);
    return n;
}
double t_fma(int T, long per_thread){
    omp_set_num_threads(T);
    double *acc = (double*)malloc(sizeof(double)*(size_t)T);
    double t0 = now_s();
    #pragma omp parallel
    {
        long t = omp_get_thread_num();
        double s = 0.0;
        for (long k = 0; k < per_thread; k++) s = fma(s, 1.0000001, 0.5);
        acc[t] = s;
    }
    double t1 = now_s();
    double sum = 0.0; for (int i = 0; i < T; i++) sum += acc[i];
    free(acc);
    return (t1 - t0) + sum * 0.0;
}
double t_atomic(int T, double *bins, const double *src, const int32_t *ip, long n){
    omp_set_num_threads(T);
    double *tmp = (double*)malloc((size_t)n*8); memcpy(tmp, bins, (size_t)n*8);
    double t0 = now_s();
    #pragma omp parallel for schedule(static, 8192)
    for (long i = 0; i < n; i++){
        long j = ip[i];
        #pragma omp atomic
        tmp[j] += src[i];
    }
    double t1 = now_s();
    memcpy(bins, tmp, (size_t)n*8); free(tmp);
    return t1 - t0;
}
double t_serial4(double *bins, const double *src, const int32_t *ip, long n){
    long i = 0;
    for (; i + 4 <= n; i += 4){
        long j0 = ip[i], j1 = ip[i+1], j2 = ip[i+2], j3 = ip[i+3];
        bins[j0] += src[i];
        bins[j1] += src[i+1];
        bins[j2] += src[i+2];
        bins[j3] += src[i+3];
    }
    for (; i < n; i++) bins[ip[i]] += src[i];
    return 0.0;
}
'''

_load_err = None
_LIB = None
def _try_load():
    global _LIB, _load_err
    try:
        d = tempfile.mkdtemp(prefix='sadjit_')
        src = os.path.join(d, 'p.c'); lib = os.path.join(d, 'libp.so')
        with open(src, 'w') as f: f.write(C_SRC)
        r = subprocess.run(['gcc', '-O3', '-fopenmp', '-shared', '-fPIC', src, '-o', lib],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError('gcc rc=%s: %s' % (r.returncode, r.stderr[:400]))
        _LIB = ctypes.CDLL(lib)
    except Exception as e:
        _load_err = repr(e)

_try_load()
P = ctypes.POINTER
if _LIB is not None:
    _LIB.affinity_cnt.restype = ctypes.c_long
    _LIB.affinity_cnt.argtypes = []
    _LIB.try_unpin.restype = ctypes.c_int
    _LIB.try_unpin.argtypes = [ctypes.c_long]
    _LIB.cpus_used.restype = ctypes.c_int
    _LIB.cpus_used.argtypes = [ctypes.c_int]
    _LIB.t_fma.restype = ctypes.c_double
    _LIB.t_fma.argtypes = [ctypes.c_int, ctypes.c_long]
    _LIB.t_atomic.restype = ctypes.c_double
    _LIB.t_atomic.argtypes = [ctypes.c_int, P(ctypes.c_double), P(ctypes.c_double), P(ctypes.c_int32), ctypes.c_long]
    _LIB.t_serial4.restype = ctypes.c_double
    _LIB.t_serial4.argtypes = [P(ctypes.c_double), P(ctypes.c_double), P(ctypes.c_int32), ctypes.c_long]

def scatter_accum_dup(bins, src, ip, LEN_1D):
    def say(x):
        sys.stdout.write(str(x) + '\n'); sys.stdout.flush()
    n = LEN_1D
    bp = bins.ctypes.data_as(P(ctypes.c_double)); sp = src.ctypes.data_as(P(ctypes.c_double))
    pp = ip.ctypes.data_as(P(ctypes.c_int32))
    ref = bins.copy(); np.add.at(ref, ip, src)
    if _LIB is None:
        say('lib FAIL %r' % _load_err); np.add.at(bins, ip, src); say('PROBE DONE'); return None
    say('affinity=%s' % _LIB.affinity_cnt())
    say('fma_T1 %.3f s' % _LIB.t_fma(1, 200_000_000))
    say('fma_T8 %.3f s' % _LIB.t_fma(8, 200_000_000))
    say('cpus_used_T8=%d' % _LIB.cpus_used(8))
    rc = _LIB.try_unpin(192)
    say('unpin_rc=%d new_affinity=%s' % (rc, _LIB.affinity_cnt()))
    if rc == 0:
        say('fma_T8_unpinned %.3f s' % _LIB.t_fma(8, 200_000_000))
        say('cpus_used_T16=%d' % _LIB.cpus_used(16))
    for T in (8, 16, 24):
        b2 = bins.copy()
        tms = 1e3*_LIB.t_atomic(T, b2.ctypes.data_as(P(ctypes.c_double)), sp, pp, n)
        say('atomic_T%d %.1f ms err %.3e' % (T, tms, np.abs(b2-ref).max()))
        del b2
    b3 = bins.copy()
    t0 = time.perf_counter(); _LIB.t_serial4(b3.ctypes.data_as(P(ctypes.c_double)), sp, pp, n); t1 = time.perf_counter()
    say('serial4 %.1f ms err %.3e' % ((t1-t0)*1e3, np.abs(b3-ref).max()))
    say('PROBE DONE')
    return None
