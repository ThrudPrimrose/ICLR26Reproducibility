#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

/* TSVC s2275: aa = aa + bb*cc (Hadamard FMA over LEN_2D x LEN_2D arrays)
 * and a  = b  + c*d  (vector FMA).
 *
 * Key optimizations:
 *  - Flatten the 2-D update into a single linear sweep in storage order
 *    (row-major), which is contiguous and lets the hardware prefetcher work.
 *  - Process both sweeps inside one OpenMP parallel region with manual
 *    static partitioning to avoid a second thread-team fork/join.
 *  - Explicit AVX-512 FMA with 2x unroll to expose instruction-level
 *    parallelism while keeping register pressure low.
 *  - Scalar tails handle the remaining < 8 elements.
 */

static inline void vec_aa_u2(double *restrict aa, const double *restrict bb, const double *restrict cc,
                             int64_t k0, int64_t k1) {
    int64_t k_end = k0 + ((k1 - k0) & ~(int64_t)1);
    int64_t kk = k0;
    for (; kk < k_end; kk += 2) {
        int64_t k = kk * 8;
        __m512d va0 = _mm512_loadu_pd(&aa[k]);
        __m512d vb0 = _mm512_loadu_pd(&bb[k]);
        __m512d vc0 = _mm512_loadu_pd(&cc[k]);
        __m512d va1 = _mm512_loadu_pd(&aa[k + 8]);
        __m512d vb1 = _mm512_loadu_pd(&bb[k + 8]);
        __m512d vc1 = _mm512_loadu_pd(&cc[k + 8]);
        va0 = _mm512_fmadd_pd(vb0, vc0, va0);
        va1 = _mm512_fmadd_pd(vb1, vc1, va1);
        _mm512_storeu_pd(&aa[k], va0);
        _mm512_storeu_pd(&aa[k + 8], va1);
    }
    if (kk < k1) {
        int64_t k = kk * 8;
        __m512d va = _mm512_loadu_pd(&aa[k]);
        __m512d vb = _mm512_loadu_pd(&bb[k]);
        __m512d vc = _mm512_loadu_pd(&cc[k]);
        va = _mm512_fmadd_pd(vb, vc, va);
        _mm512_storeu_pd(&aa[k], va);
    }
}

static inline void vec_a_u2(double *restrict a, const double *restrict b, const double *restrict c,
                            const double *restrict d, int64_t i0, int64_t i1) {
    int64_t i_end = i0 + ((i1 - i0) & ~(int64_t)1);
    int64_t ii = i0;
    for (; ii < i_end; ii += 2) {
        int64_t i = ii * 8;
        __m512d va0 = _mm512_loadu_pd(&a[i]);
        __m512d vb0 = _mm512_loadu_pd(&b[i]);
        __m512d vc0 = _mm512_loadu_pd(&c[i]);
        __m512d vd0 = _mm512_loadu_pd(&d[i]);
        __m512d va1 = _mm512_loadu_pd(&a[i + 8]);
        __m512d vb1 = _mm512_loadu_pd(&b[i + 8]);
        __m512d vc1 = _mm512_loadu_pd(&c[i + 8]);
        __m512d vd1 = _mm512_loadu_pd(&d[i + 8]);
        va0 = _mm512_fmadd_pd(vc0, vd0, vb0);
        va1 = _mm512_fmadd_pd(vc1, vd1, vb1);
        _mm512_storeu_pd(&a[i], va0);
        _mm512_storeu_pd(&a[i + 8], va1);
    }
    if (ii < i1) {
        int64_t i = ii * 8;
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vb = _mm512_loadu_pd(&b[i]);
        __m512d vc = _mm512_loadu_pd(&c[i]);
        __m512d vd = _mm512_loadu_pd(&d[i]);
        va = _mm512_fmadd_pd(vc, vd, vb);
        _mm512_storeu_pd(&a[i], va);
    }
}

void tsvc_2_s2275_fp64(double *restrict a, double *restrict aa, const double *restrict b, const double *restrict bb,
                       const double *restrict c, const double *restrict cc, const double *restrict d,
                       const int64_t LEN_2D) {

    const int64_t n2 = LEN_2D * LEN_2D;
    const int64_t n2_vec = n2 / 8;
    const int64_t n_vec = LEN_2D / 8;

#pragma omp parallel
{
    const int64_t tid = omp_get_thread_num();
    const int64_t nt = omp_get_num_threads();

    int64_t k0 = (n2_vec * tid) / nt;
    int64_t k1 = (n2_vec * (tid + 1)) / nt;
    vec_aa_u2(aa, bb, cc, k0, k1);

    int64_t i0 = (n_vec * tid) / nt;
    int64_t i1 = (n_vec * (tid + 1)) / nt;
    vec_a_u2(a, b, c, d, i0, i1);
}

    for (int64_t k = n2_vec * 8; k < n2; ++k) {
        aa[k] = aa[k] + bb[k] * cc[k];
    }

    for (int64_t i = n_vec * 8; i < LEN_2D; ++i) {
        a[i] = b[i] + c[i] * d[i];
    }
}
