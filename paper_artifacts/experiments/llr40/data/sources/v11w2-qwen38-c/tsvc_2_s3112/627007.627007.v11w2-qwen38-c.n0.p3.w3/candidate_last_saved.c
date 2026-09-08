#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>

static void scan_block(const double *restrict a, double *restrict b, int64_t n, double carry) {
    const __m512i idx1 = _mm512_set_epi64(6,5,4,3,2,1,0,0);
    const __m512i idx2 = _mm512_set_epi64(5,4,3,2,1,0,1,0);
    const __m512i idx4 = _mm512_set_epi64(3,2,1,0,3,2,1,0);
    const __m512i idx7 = _mm512_set1_epi64(7);
    const __mmask8 m1 = 0xFE, m2 = 0xFC, m4 = 0xF0;
    int64_t i = 0;
    __m512d c = _mm512_set1_pd(carry);
    for (; i + 8 <= n; i += 8) {
        __m512d s = _mm512_loadu_pd(a + i);
        s = _mm512_mask_add_pd(s, m1, s, _mm512_permutexvar_pd(idx1, s));
        s = _mm512_mask_add_pd(s, m2, s, _mm512_permutexvar_pd(idx2, s));
        s = _mm512_mask_add_pd(s, m4, s, _mm512_permutexvar_pd(idx4, s));
        __m512d o = _mm512_add_pd(s, c);
        c = _mm512_add_pd(c, _mm512_permutexvar_pd(idx7, s));
        _mm512_stream_pd(b + i, o);
    }
    double sc = _mm512_cvtsd_f64(c);
    for (; i < n; ++i) { sc += a[i]; b[i] = sc; }
}

static double reduce_block(const double *restrict a, int64_t n) {
    __m512d s0 =_mm512_setzero_pd(), s1 =_mm512_setzero_pd(), s2 =_mm512_setzero_pd(), s3 =_mm512_setzero_pd();
    __m512d s4 =_mm512_setzero_pd(), s5 =_mm512_setzero_pd(), s6 =_mm512_setzero_pd(), s7 =_mm512_setzero_pd();
    __m512d s8 =_mm512_setzero_pd(), s9 =_mm512_setzero_pd(), s10=_mm512_setzero_pd(), s11=_mm512_setzero_pd();
    __m512d s12=_mm512_setzero_pd(), s13=_mm512_setzero_pd(), s14=_mm512_setzero_pd(), s15=_mm512_setzero_pd();
    int64_t i = 0;
    for (; i + 128 <= n; i += 128) {
        _mm_prefetch((const char*)(a+i+2048), _MM_HINT_T0);
        s0  = _mm512_add_pd(s0 , _mm512_loadu_pd(a+i));
        s1  = _mm512_add_pd(s1 , _mm512_loadu_pd(a+i+8));
        s2  = _mm512_add_pd(s2 , _mm512_loadu_pd(a+i+16));
        s3  = _mm512_add_pd(s3 , _mm512_loadu_pd(a+i+24));
        s4  = _mm512_add_pd(s4 , _mm512_loadu_pd(a+i+32));
        s5  = _mm512_add_pd(s5 , _mm512_loadu_pd(a+i+40));
        s6  = _mm512_add_pd(s6 , _mm512_loadu_pd(a+i+48));
        s7  = _mm512_add_pd(s7 , _mm512_loadu_pd(a+i+56));
        s8  = _mm512_add_pd(s8 , _mm512_loadu_pd(a+i+64));
        s9  = _mm512_add_pd(s9 , _mm512_loadu_pd(a+i+72));
        s10 = _mm512_add_pd(s10, _mm512_loadu_pd(a+i+80));
        s11 = _mm512_add_pd(s11, _mm512_loadu_pd(a+i+88));
        s12 = _mm512_add_pd(s12, _mm512_loadu_pd(a+i+96));
        s13 = _mm512_add_pd(s13, _mm512_loadu_pd(a+i+104));
        s14 = _mm512_add_pd(s14, _mm512_loadu_pd(a+i+112));
        s15 = _mm512_add_pd(s15, _mm512_loadu_pd(a+i+120));
    }
    __m512d t0 = _mm512_add_pd(_mm512_add_pd(s0,s1), _mm512_add_pd(s2,s3));
    __m512d t1 = _mm512_add_pd(_mm512_add_pd(s4,s5), _mm512_add_pd(s6,s7));
    __m512d t2 = _mm512_add_pd(_mm512_add_pd(s8,s9), _mm512_add_pd(s10,s11));
    __m512d t3 = _mm512_add_pd(_mm512_add_pd(s12,s13), _mm512_add_pd(s14,s15));
    __m512d t = _mm512_add_pd(_mm512_add_pd(t0,t1), _mm512_add_pd(t2,t3));
    double r[8]; _mm512_storeu_pd(r, t);
    double acc = 0.0;
    for (int j=0;j<8;++j) acc += r[j];
    for (; i < n; ++i) acc += a[i];
    return acc;
}

static void run2phase(const double *a, double *b, int64_t N, int nt, double *blksum, double *tp) {
    double t0 = omp_get_wtime();
    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num(), ntid = omp_get_num_threads();
        int64_t start = ((N * (int64_t)tid) / ntid) & ~7LL;
        int64_t end   = (tid+1 < ntid) ? (((N*(int64_t)(tid+1))/ntid) & ~7LL) : N;
        blksum[tid] = reduce_block(a + start, end - start);
    }
    double t1 = omp_get_wtime();
    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num(), ntid = omp_get_num_threads();
        int64_t start = ((N * (int64_t)tid) / ntid) & ~7LL;
        int64_t end   = (tid+1 < ntid) ? (((N*(int64_t)(tid+1))/ntid) & ~7LL) : N;
        double carry = 0.0;
        for (int k = 0; k < tid; ++k) carry += blksum[k];
        scan_block(a + start, b + start, end - start, carry);
    }
    double t2 = omp_get_wtime();
    tp[0] = t1 - t0; tp[1] = t2 - t1;
}

static double read_probe(const double *a, int64_t N, int nt) {
    double t0 = omp_get_wtime();
    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num(), ntid = omp_get_num_threads();
        int64_t start = ((N * (int64_t)tid) / ntid) & ~7LL;
        int64_t end   = (tid+1 < ntid) ? (((N*(int64_t)(tid+1))/ntid) & ~7LL) : N;
        volatile double acc = 0.0;
        const double *p = a + start;
        for (int64_t i = start, e = end; i < e; i += 64) {
            _mm_prefetch((const char*)(p + i - start + 2048), _MM_HINT_T0);
            acc += p[i - start] + p[i - start + 1] + p[i - start + 2] + p[i - start + 3];
        }
    }
    double t1 = omp_get_wtime();
    return t1 - t0;
}

static double write_probe(double *b, int64_t N, int nt) {
    double t0 = omp_get_wtime();
    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num(), ntid = omp_get_num_threads();
        int64_t start = ((N * (int64_t)tid) / ntid) & ~7LL;
        int64_t end   = (tid+1 < ntid) ? (((N*(int64_t)(tid+1))/ntid) & ~7LL) : N;
        __m512d v = _mm512_set1_pd(1.5);
        for (int64_t i = start; i + 8 <= end; i += 8) _mm512_stream_pd(b + i, v);
    }
    double t1 = omp_get_wtime();
    return t1 - t0;
}

void tsvc_2_s3112_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
    const int64_t N = LEN_1D;
    if (N <= 0) return;
    const int nt = omp_get_max_threads();
    double *blksum = (double*)malloc(sizeof(double)*nt);
    double tp[2];
    double tr = read_probe(a, N, nt);
    double tw = write_probe(b, N, nt);
    run2phase(a, b, N, nt, blksum, tp);
    printf("read %.2f ms (%.1f GB/s)  writeNT %.2f ms (%.1f GB/s)  reduce %.2f ms (%.1f GB/s)  scan %.2f ms (%.1f GB/s)  total2ph %.2f ms\n",
        tr*1e3, N*8.0/1e9/tr, tw*1e3, N*8.0/1e9/tw, tp[0]*1e3, N*8.0/1e9/tp[0], tp[1]*1e3, N*16.0/1e9/tp[1], (tp[0]+tp[1])*1e3);
    fflush(stdout);
    free(blksum);
}
