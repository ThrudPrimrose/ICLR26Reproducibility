/* Optimized FDTD 2D (PolyBench). Same C-ABI and numerical semantics as the
 * reference; restructured as two dependency-safe phases per time step:
 *   phase A: ex, ey from OLD hz   (parallel over rows)
 *   phase B: hz from NEW ex, ey   (parallel over rows)
 * with streaming AVX2 (256-bit) SIMD and OpenMP over contiguous row blocks.
 */
#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#if defined(__AVX2__)
#define FDTD_SIMD 1
#else
#define FDTD_SIMD 0
#endif

static void row_ey_ex(
    const double *restrict hz_prev, const double *restrict hz_cur,
    double *restrict ey_row, double *restrict ex_row,
    int64_t NY, double ce, double cx)
{
    /* ey_row -= ce * (hz_cur - hz_prev)  (all columns)
     * ex_row[1:NY] -= cx * (hz_cur[j] - hz_cur[j-1]) */
#if FDTD_SIMD
    const __m256d vce = _mm256_set1_pd(ce);
    const __m256d vcx = _mm256_set1_pd(cx);
    int64_t j = 0;
    for (; j + 4 <= NY; j += 4) {
        __m256d ha = _mm256_loadu_pd(hz_prev + j);
        __m256d hb = _mm256_loadu_pd(hz_cur + j);
        __m256d ea = _mm256_loadu_pd(ey_row + j);
        _mm256_storeu_pd(ey_row + j, _mm256_fnmadd_pd(hb, vce, _mm256_fmadd_pd(ha, vce, ea)));
    }
    for (; j < NY; ++j)
        ey_row[j] -= ce * (hz_cur[j] - hz_prev[j]);
    for (j = 1; j + 4 <= NY; j += 4) {
        __m256d lo = _mm256_loadu_pd(hz_cur + j - 1);
        __m256d hi = _mm256_loadu_pd(hz_cur + j);
        __m256d xa = _mm256_loadu_pd(ex_row + j);
        _mm256_storeu_pd(ex_row + j, _mm256_fnmadd_pd(_mm256_sub_pd(hi, lo), vcx, xa));
    }
    for (; j < NY; ++j)
        ex_row[j] -= cx * (hz_cur[j] - hz_cur[j - 1]);
#else
    int64_t j;
    for (j = 0; j < NY; ++j)
        ey_row[j] -= ce * (hz_cur[j] - hz_prev[j]);
    for (j = 1; j < NY; ++j)
        ex_row[j] -= cx * (hz_cur[j] - hz_cur[j - 1]);
#endif
}

static void row_ey_ex_firstrow(
    const double *restrict hz_cur,
    double *restrict ey_row, double *restrict ex_row,
    int64_t NY, double fict_t, double cx)
{
#if FDTD_SIMD
    const __m256d vf = _mm256_set1_pd(fict_t);
    const __m256d vcx = _mm256_set1_pd(cx);
    int64_t j = 0;
    for (; j + 4 <= NY; j += 4)
        _mm256_storeu_pd(ey_row + j, vf);
    for (; j < NY; ++j)
        ey_row[j] = fict_t;
    for (j = 1; j + 4 <= NY; j += 4) {
        __m256d lo = _mm256_loadu_pd(hz_cur + j - 1);
        __m256d hi = _mm256_loadu_pd(hz_cur + j);
        __m256d xa = _mm256_loadu_pd(ex_row + j);
        _mm256_storeu_pd(ex_row + j, _mm256_fnmadd_pd(_mm256_sub_pd(hi, lo), vcx, xa));
    }
    for (; j < NY; ++j)
        ex_row[j] -= cx * (hz_cur[j] - hz_cur[j - 1]);
#else
    int64_t j;
    for (j = 0; j < NY; ++j)
        ey_row[j] = fict_t;
    for (j = 1; j < NY; ++j)
        ex_row[j] -= cx * (hz_cur[j] - hz_cur[j - 1]);
#endif
}

static void row_hz(
    const double *restrict ex_row, const double *restrict ey_row,
    const double *restrict ey_next, double *restrict hz_row,
    int64_t NY, double ch)
{
    /* hz_row[0:NY-1] -= ch * ((ex_row[j+1] - ex_row[j]) + (ey_next[j] - ey_row[j])) */
#if FDTD_SIMD
    const __m256d vch = _mm256_set1_pd(ch);
    int64_t j = 0;
    for (; j + 4 <= NY - 1; j += 4) {
        __m256d xl = _mm256_loadu_pd(ex_row + j);
        __m256d xh = _mm256_loadu_pd(ex_row + j + 1);
        __m256d ya = _mm256_loadu_pd(ey_row + j);
        __m256d yb = _mm256_loadu_pd(ey_next + j);
        __m256d t = _mm256_sub_pd(_mm256_add_pd(_mm256_sub_pd(xh, xl), yb), ya);
        __m256d ha = _mm256_loadu_pd(hz_row + j);
        _mm256_storeu_pd(hz_row + j, _mm256_fnmadd_pd(t, vch, ha));
    }
    for (; j < NY - 1; ++j)
        hz_row[j] -= ch * (((ex_row[j + 1] - ex_row[j]) + ey_next[j]) - ey_row[j]);
#else
    int64_t j;
    for (j = 0; j < NY - 1; ++j)
        hz_row[j] -= ch * (((ex_row[j + 1] - ex_row[j]) + ey_next[j]) - ey_row[j]);
#endif
}

void fdtd_2d_fp64(double *restrict ex, double *restrict ey, const double *restrict fict,
                  double *restrict hz, const int64_t NX, const int64_t NY,
                  const int64_t TMAX, const double ex_courant, const double ey_courant,
                  const double hz_courant)
{
    if (NX <= 0 || NY <= 0 || TMAX <= 0)
        return;

#pragma omp parallel
    {
        const int64_t n = NX;
        for (int64_t t = 0; t < TMAX; ++t) {
            /* --- phase A: ey (row0 = fict), ex  -- read old hz --- */
#pragma omp for schedule(static)
            for (int64_t i = 0; i < n; ++i) {
                double *e = ey + i * NY;
                double *x = ex + i * NY;
                if (i == 0)
                    row_ey_ex_firstrow(hz, e, x, NY, fict[t], ex_courant);
                else
                    row_ey_ex(hz + (i - 1) * NY, hz + i * NY, e, x, NY, ey_courant, ex_courant);
            }
            /* --- phase B: hz  -- read new ex/ey, hz rows 0..NX-2 --- */
#pragma omp for schedule(static)
            for (int64_t i = 0; i < n; ++i) {
                if (i >= NX - 1)
                    continue;
                row_hz(ex + i * NY, ey + i * NY, ey + (i + 1) * NY, hz + i * NY, NY, hz_courant);
            }
        }
    }
}
