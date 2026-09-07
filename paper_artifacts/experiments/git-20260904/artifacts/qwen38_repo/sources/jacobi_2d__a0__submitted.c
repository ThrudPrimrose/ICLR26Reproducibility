// 2-D Jacobi 5-point stencil, hand-tuned AVX-512, OpenMP row-parallel.
//
// Bit-exactness: each output keeps the reference association
//   0.2 * ((((c + l) + r) + d) + u)
// i.e. five loads, four adds in reference order, one standalone trailing
// multiply -- FMA contraction can never change the result.
//
// One team for the whole call; the timestep loop is a sequential
// recurrence. Each sweep is an `omp for` over rows (static schedule:
// contiguous chunks, halo rows hit the LLC).
//
// Inner loop: 4 independent 8-wide SIMD blocks per iteration, kept in
// four distinct temporaries so the out-of-order window overlaps their
// 5-load / 4-add chains (register reuse across blocks serializes them).
#include <stdint.h>
#include <immintrin.h>

static inline void row_stencil(const double *restrict au, const double *restrict ac,
                               const double *restrict ad, double *restrict bc, int64_t n)
{
    const __m512d c02 = _mm512_set1_pd(0.2);
    int64_t j = 1;
    const int64_t core_lo = n - 31;
    if (n >= 32) {
        for (; j < 8; ++j)
            bc[j] = 0.2 * ((((ac[j] + ac[j-1]) + ac[j+1]) + ad[j]) + au[j]);
        for (; j <= core_lo; j += 32) {
            /* block 0: c+l */
            __m512d x0 = _mm512_add_pd(_mm512_loadu_pd(ac + j),    _mm512_loadu_pd(ac + j - 1));
            __m512d x1 = _mm512_add_pd(_mm512_loadu_pd(ac + j + 8), _mm512_loadu_pd(ac + j + 7));
            __m512d x2 = _mm512_add_pd(_mm512_loadu_pd(ac + j + 16),_mm512_loadu_pd(ac + j + 15));
            __m512d x3 = _mm512_add_pd(_mm512_loadu_pd(ac + j + 24),_mm512_loadu_pd(ac + j + 23));
            /* + r */
            x0 = _mm512_add_pd(x0, _mm512_loadu_pd(ac + j + 1));
            x1 = _mm512_add_pd(x1, _mm512_loadu_pd(ac + j + 9));
            x2 = _mm512_add_pd(x2, _mm512_loadu_pd(ac + j + 17));
            x3 = _mm512_add_pd(x3, _mm512_loadu_pd(ac + j + 25));
            /* + d */
            x0 = _mm512_add_pd(x0, _mm512_loadu_pd(ad + j));
            x1 = _mm512_add_pd(x1, _mm512_loadu_pd(ad + j + 8));
            x2 = _mm512_add_pd(x2, _mm512_loadu_pd(ad + j + 16));
            x3 = _mm512_add_pd(x3, _mm512_loadu_pd(ad + j + 24));
            /* + u */
            x0 = _mm512_add_pd(x0, _mm512_loadu_pd(au + j));
            x1 = _mm512_add_pd(x1, _mm512_loadu_pd(au + j + 8));
            x2 = _mm512_add_pd(x2, _mm512_loadu_pd(au + j + 16));
            x3 = _mm512_add_pd(x3, _mm512_loadu_pd(au + j + 24));
            /* out = 0.2 * (sum) */
            _mm512_storeu_pd(bc + j,     _mm512_mul_pd(x0, c02));
            _mm512_storeu_pd(bc + j + 8, _mm512_mul_pd(x1, c02));
            _mm512_storeu_pd(bc + j + 16, _mm512_mul_pd(x2, c02));
            _mm512_storeu_pd(bc + j + 24, _mm512_mul_pd(x3, c02));
        }
    }
    for (; j <= n; ++j)
        bc[j] = 0.2 * ((((ac[j] + ac[j-1]) + ac[j+1]) + ad[j]) + au[j]);
}

void jacobi_2d_fp64(double *restrict A, double *restrict B, const int64_t N, const int64_t TSTEPS)
{
    const int64_t NI = N - 1;
    if (TSTEPS <= 0 || NI <= 1) return;

    #pragma omp parallel
    {
        for (int64_t t = 0; t < TSTEPS; ++t) {
            #pragma omp for schedule(static)
            for (int64_t i = 1; i < NI; ++i)
                row_stencil(A + (i - 1) * N, A + i * N, A + (i + 1) * N, B + i * N, N - 2);
            #pragma omp for schedule(static)
            for (int64_t i = 1; i < NI; ++i)
                row_stencil(B + (i - 1) * N, B + i * N, B + (i + 1) * N, A + i * N, N - 2);
        }
    }
}
