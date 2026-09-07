#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

/* v8: hand AVX-512, 16 pts/iter, 10 zmm loads/16 pts (center row loaded once,
   k-1/k+1 derived by register shifts). Plain (non-FMA) ops replicating the
   reference tree exactly: out = ((alpha*((pip-2c)+pim) + alpha*((pjp-2c)+pjm))
                                 + alpha*((cn-2c)+cp)) + c                      */
static void row1(const double *pim, const double *pip, const double *pjm,
                 const double *pjp, const double *pc, double *bi,
                 int64_t k, double alpha);

static inline void row7x16(const double *__restrict pim, const double *__restrict pip,
                           const double *__restrict pjm, const double *__restrict pjp,
                           const double *__restrict pc, double *__restrict bi,
                           int64_t N, double alpha) {
    const __m512d two = _mm512_set1_pd(2.0);
    const __m512d al  = _mm512_set1_pd(alpha);
    const __m512i id_shl = _mm512_setr_epi64(1, 2, 3, 4, 5, 6, 7, 7); /* [x1..x7,x7] */
    const __m512i id_shr = _mm512_setr_epi64(7, 0, 1, 2, 3, 4, 5, 6); /* [x7,x0..x6] */
    const __m512i id_l0  = _mm512_set1_epi64(0);
    int64_t k = 1;
    const int64_t kmax = N - 1; /* valid k: 1..N-2 */
    for (; k + 16 <= kmax; k += 16) {
        __m512d ip0 = _mm512_loadu_pd(pip + k), ip1 = _mm512_loadu_pd(pip + k + 8);
        __m512d im0 = _mm512_loadu_pd(pim + k), im1 = _mm512_loadu_pd(pim + k + 8);
        __m512d jp0 = _mm512_loadu_pd(pjp + k), jp1 = _mm512_loadu_pd(pjp + k + 8);
        __m512d jm0 = _mm512_loadu_pd(pjm + k), jm1 = _mm512_loadu_pd(pjm + k + 8);
        __m512d c0  = _mm512_loadu_pd(pc + k), c1  = _mm512_loadu_pd(pc + k + 8);

        __m512d cn0 = _mm512_mask_blend_pd(0x80, _mm512_permutexvar_pd(id_shl, c0), _mm512_permutexvar_pd(id_l0, c1));
        __m512d cp0 = _mm512_mask_blend_pd(0x01, _mm512_permutexvar_pd(id_shr, c0), _mm512_set1_pd(pc[k - 1]));
        __m512d cn1 = _mm512_mask_blend_pd(0x80, _mm512_permutexvar_pd(id_shl, c1), _mm512_set1_pd(pc[k + 16]));
        __m512d cp1 = _mm512_mask_blend_pd(0x01, _mm512_permutexvar_pd(id_shr, c1), _mm512_permutexvar_pd(id_shr, c0));

        __m512d t0 = _mm512_mul_pd(two, c0);
        __m512d o0 = _mm512_add_pd(
            _mm512_add_pd(
                _mm512_add_pd(_mm512_mul_pd(al, _mm512_add_pd(_mm512_sub_pd(ip0, t0), im0)),
                              _mm512_mul_pd(al, _mm512_add_pd(_mm512_sub_pd(jp0, t0), jm0))),
                _mm512_mul_pd(al, _mm512_add_pd(_mm512_sub_pd(cn0, t0), cp0))),
            c0);
        _mm512_storeu_pd(bi + k, o0);

        __m512d t1 = _mm512_mul_pd(two, c1);
        __m512d o1 = _mm512_add_pd(
            _mm512_add_pd(
                _mm512_add_pd(_mm512_mul_pd(al, _mm512_add_pd(_mm512_sub_pd(ip1, t1), im1)),
                              _mm512_mul_pd(al, _mm512_add_pd(_mm512_sub_pd(jp1, t1), jm1))),
                _mm512_mul_pd(al, _mm512_add_pd(_mm512_sub_pd(cn1, t1), cp1))),
            c1);
        _mm512_storeu_pd(bi + k + 8, o1);
    }
    for (; k < kmax; ++k)
        row1(pim, pip, pjm, pjp, pc, bi, k, alpha);
}

/* scalar fallback: no FP contraction allowed (bit-exact vs numpy for any alpha) */
#pragma GCC optimize ("fp-contract=off")
__attribute__((noinline))
static void row1(const double *pim, const double *pip, const double *pjm,
                 const double *pjp, const double *pc, double *bi,
                 int64_t k, double alpha) {
    double t = 2.0 * pc[k];
    double ac = alpha * (pip[k] - t + pim[k]);
    double aj = alpha * (pjp[k] - t + pjm[k]);
    double ak = alpha * (pc[k + 1] - t + pc[k - 1]);
    bi[k] = ((ac + aj) + ak) + pc[k];
}

static inline void sweep(double *__restrict C, const double *__restrict S,
                         int64_t N, double alpha) {
    #pragma omp for schedule(static)
    for (int64_t i = 1; i < N - 1; ++i) {
        const int64_t iN = i * N * N;
        for (int64_t j = 1; j < N - 1; ++j) {
            row7x16(S + (i - 1) * N * N + j * N, S + (i + 1) * N * N + j * N,
                    S + iN + (j - 1) * N, S + iN + (j + 1) * N,
                    S + iN + j * N, C + iN + j * N, N, alpha);
        }
    }
}

void heat_3d_fp64(double *restrict A, double *restrict B, const int64_t N,
                  const int64_t TSTEPS, const double alpha) {
    #pragma omp parallel
    for (int64_t t = 0; t < TSTEPS; ++t) {
        sweep(B, A, N, alpha);
        sweep(A, B, N, alpha);
    }
}
