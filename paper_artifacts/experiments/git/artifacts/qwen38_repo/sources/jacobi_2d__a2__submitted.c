/* Optimized 2-D Jacobi stencil (5-point), double precision.
 *
 * Strategy:
 *  - The TSTEPS loop is a recurrence (A_t -> B_t -> A_{t+1}), so time steps
 *    are serial; the interior of each sweep is embarrassingly parallel in the
 *    row index (dst row i only reads src rows i-1..i+1 and writes dst row i).
 *  - One OpenMP team for the whole call; each thread owns a contiguous band
 *    of rows for every time step, so its 3-row working set stays hot in L2
 *    across the entire run and thread-pool startup happens exactly once.
 *  - The inner column loop is SIMD (AVX2, 4 doubles/lane) with exactly the
 *    same per-element operation order as the reference:
 *        0.2 * ((((c + l) + r) + d) + u)
 *    so results are bit-identical to the NumPy oracle.
 */
#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

static inline void sweep_row(const double *restrict r0,
                             const double *restrict r1,
                             const double *restrict r2,
                             double *restrict d, int64_t N)
{
    int64_t j = 1;
    const __m256d v02 = _mm256_set1_pd(0.2);
    for (; j + 4 <= N - 1; j += 4) {
        __m256d vc = _mm256_loadu_pd(r1 + j);
        __m256d vl = _mm256_loadu_pd(r1 + j - 1);
        __m256d vr = _mm256_loadu_pd(r1 + j + 1);
        __m256d vd = _mm256_loadu_pd(r2 + j);
        __m256d vu = _mm256_loadu_pd(r0 + j);
        __m256d s = _mm256_add_pd(vc, vl);
        s = _mm256_add_pd(s, vr);
        s = _mm256_add_pd(s, vd);
        s = _mm256_add_pd(s, vu);
        _mm256_storeu_pd(d + j, _mm256_mul_pd(s, v02));
    }
    for (; j < N - 1; ++j)
        d[j] = 0.2 * (r1[j] + r1[j - 1] + r1[j + 1] + r2[j] + r0[j]);
}

void jacobi_2d_fp64(double *restrict A, double *restrict B, const int64_t N, const int64_t TSTEPS) {
    if (N < 3 || TSTEPS <= 0)
        return;
    const int64_t nint = N - 2; /* interior rows: i = 1 .. N-2 */

    #pragma omp parallel
    {
        const int64_t tid = (int64_t)omp_get_thread_num();
        const int64_t nt  = (int64_t)omp_get_num_threads();
        const int64_t l0 = (nint * tid) / nt;
        const int64_t l1 = (nint * (tid + 1)) / nt;

        for (int64_t t = 0; t < TSTEPS; ++t) {
            /* sweep A -> B */
            for (int64_t l = l0; l < l1; ++l) {
                const int64_t i = l + 1;
                sweep_row(A + (i - 1) * N, A + i * N, A + (i + 1) * N, B + i * N, N);
            }
            #pragma omp barrier
            /* sweep B -> A */
            for (int64_t l = l0; l < l1; ++l) {
                const int64_t i = l + 1;
                sweep_row(B + (i - 1) * N, B + i * N, B + (i + 1) * N, A + i * N, N);
            }
            /* before the next sweep reads the rows neighbors just wrote */
            #pragma omp barrier
        }
    }
}
