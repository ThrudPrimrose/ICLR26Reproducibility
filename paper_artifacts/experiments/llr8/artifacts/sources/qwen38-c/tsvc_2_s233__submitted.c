/* TSVC s233 -- two independent 2-D prefix scans.
 *
 *   aa[j,i] = aa[j-1,i] + cc[j,i]   chain along j  -> parallel across i (columns)
 *   bb[j,i] = bb[j,i-1] + cc[j,i]   chain along i  -> parallel across j (rows)
 *
 * Pass A interleaves VEC consecutive COLUMNS in one SIMD register: the j-chain
 * is then VEC independent chains, and each step is a unit-stride 64B load of cc
 * (row j) + vector add + unit-stride store to aa (row j).
 * Pass B interleaves VEC consecutive ROWS: the i-chain is VEC independent
 * chains, each step gathers/scatters VEC strided doubles.
 */
#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_s233_fp64(double *aa, double *bb, double *cc,
                      int64_t LEN_2D, void *workspace, int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;
    const int64_t N = LEN_2D;
    if (N <= 8)
        return;

    const int64_t n = N - 8; /* trip count of both scans */

#if defined(__AVX512F__)
    const int64_t VEC = 8;
    const int64_t G = n / VEC;

    /* ---- pass A: aa chain down columns; lanes = VEC consecutive columns ---- */
    #pragma omp parallel for schedule(static)
    for (int64_t g = 0; g < G; g++) {
        const int64_t i0 = 8 + g * VEC;
        __m512d carry = _mm512_loadu_pd(aa + 7 * N + i0);
        for (int64_t j = 8; j < N; j++) {
            carry = _mm512_add_pd(carry, _mm512_loadu_pd(cc + j * N + i0));
            _mm512_storeu_pd(aa + j * N + i0, carry);
        }
    }

    /* ---- pass B: bb chain across columns; lanes = VEC consecutive rows ---- */
    #pragma omp parallel for schedule(static)
    for (int64_t g = 0; g < G; g++) {
        const int64_t j0 = 8 + g * VEC;
        __m512i idx = _mm512_setr_epi64(
            ((j0 + 0) * N + 7) << 3, ((j0 + 1) * N + 7) << 3,
            ((j0 + 2) * N + 7) << 3, ((j0 + 3) * N + 7) << 3,
            ((j0 + 4) * N + 7) << 3, ((j0 + 5) * N + 7) << 3,
            ((j0 + 6) * N + 7) << 3, ((j0 + 7) * N + 7) << 3);
        __m512d carry = _mm512_i64gather_pd(idx, bb, 1);
        for (int64_t i = 8; i < N; i++) {
            idx = _mm512_add_epi64(idx, _mm512_set1_epi64(8));
            carry = _mm512_add_pd(carry, _mm512_i64gather_pd(idx, cc, 1));
            _mm512_i64scatter_pd(bb, idx, carry, 1);
        }
    }

#elif defined(__AVX2__)
    const int64_t VEC = 4;
    const int64_t G = n / VEC;

    #pragma omp parallel for schedule(static)
    for (int64_t g = 0; g < G; g++) {
        const int64_t i0 = 8 + g * VEC;
        __m256d carry = _mm256_loadu_pd(aa + 7 * N + i0);
        for (int64_t j = 8; j < N; j++) {
            carry = _mm256_add_pd(carry, _mm256_loadu_pd(cc + j * N + i0));
            _mm256_storeu_pd(aa + j * N + i0, carry);
        }
    }

    #pragma omp parallel for schedule(static)
    for (int64_t g = 0; g < G; g++) {
        const int64_t j0 = 8 + g * VEC;
        __m256i idx = _mm256_setr_epi64(
            ((j0 + 0) * N + 7) << 3, ((j0 + 1) * N + 7) << 3,
            ((j0 + 2) * N + 7) << 3, ((j0 + 3) * N + 7) << 3);
        __m256d carry = _mm256_i64gather_pd(idx, bb, 1);
        for (int64_t i = 8; i < N; i++) {
            idx = _mm256_add_epi64(idx, _mm256_set1_epi64(8));
            carry = _mm256_add_pd(carry, _mm256_i64gather_pd(idx, cc, 1));
            _mm256_i64scatter_pd(idx, bb, 1, carry);
        }
    }

#else
    const int64_t VEC = 1;
    const int64_t G = n / VEC;

    #pragma omp parallel for schedule(static)
    for (int64_t i = 8; i < N; i++) {
        double carry = aa[7 * N + i];
        for (int64_t j = 8; j < N; j++) {
            carry += cc[j * N + i];
            aa[j * N + i] = carry;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int64_t j = 8; j < N; j++) {
        double carry = bb[j * N + 7];
        for (int64_t i = 8; i < N; i++) {
            carry += cc[j * N + i];
            bb[j * N + i] = carry;
        }
    }
#endif

    /* ---- trailing remainder (fewer than VEC columns / rows) ---- */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 8 + G * VEC; i < N; i++) {
        double carry = aa[7 * N + i];
        for (int64_t j = 8; j < N; j++) {
            carry += cc[j * N + i];
            aa[j * N + i] = carry;
        }
    }
    #pragma omp parallel for schedule(static)
    for (int64_t j = 8 + G * VEC; j < N; j++) {
        double carry = bb[j * N + 7];
        for (int64_t i = 8; i < N; i++) {
            carry += cc[j * N + i];
            bb[j * N + i] = carry;
        }
    }
}
