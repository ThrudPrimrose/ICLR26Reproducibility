// Optimized covariance_fp64: column-mean centering + blocked AVX-512/AVX2 GEMM
// (cov = Dc^T Dc / (float_n - 1), Dc = centered data). Upper triangle only + mirror.
#define _GNU_SOURCE
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

#define SIMDD 8   // doubles per vector on AVX-512

#if defined(__AVX512F__)
typedef __m512d v8d;
#define VZERO() _mm512_setzero_pd()
#define VADD(a,b) _mm512_add_pd((a),(b))
#define VMUL(a,b) _mm512_mul_pd((a),(b))
#define VFMA(a,b,c) _mm512_fmadd_pd((a),(b),(c))
#define VONE(x) _mm512_set1_pd((x))
#define VLOAD(p) _mm512_loadu_pd((p))
#define VSTORE(p,v) _mm512_storeu_pd((p),(v))
#define VEXTRACT(v,i) _mm512_extract_pd((v),(i))
#endif

// microkernel register tile: MI i-rows x NJ j-vectors
#define MI 6
#define NJ 2
#define JC (NJ * SIMDD)

void covariance_fp64(double *restrict cov, double *restrict data, const int64_t M, const int64_t N, const double float_n)
#if defined(__AVX512F__)
{
    const int64_t nthreads = omp_get_max_threads();
    double *colsum = (double *)malloc((size_t)nthreads * (size_t)M * sizeof(double));
    if (!colsum) {
        // tiny fallback: fully serial
        double *m = (double *)malloc((size_t)M * sizeof(double));
        for (int64_t i = 0; i < M; i++) m[i] = 0.0;
        for (int64_t n = 0; n < N; n++)
            for (int64_t i = 0; i < M; i++) m[i] += data[n * M + i];
        for (int64_t n = 0; n < N; n++)
            for (int64_t i = 0; i < M; i++) data[n * M + i] -= m[i] / (double)N;
        for (int64_t i = 0; i < M; i++)
            for (int64_t j = i; j < M; j++) {
                double s = 0.0;
                for (int64_t k = 0; k < N; k++) s += data[k * M + i] * data[k * M + j];
                cov[i * M + j] = s / (float_n - 1.0);
                if (j != i) cov[j * M + i] = cov[i * M + j];
            }
        free(m);
        return;
    }
    memset(colsum, 0, (size_t)nthreads * (size_t)M * sizeof(double));

    // ---- pass 1: column sums (row-parallel for locality) ----
    #pragma omp parallel
    {
        int64_t t = omp_get_thread_num();
        double *pc = colsum + t * M;
        #pragma omp for
        for (int64_t n = 0; n < N; n++) {
            const double *row = data + n * M;
            #pragma omp simd
            for (int64_t i = 0; i < M; i++) pc[i] += row[i];
        }
    }
    double *mean = colsum; // thread 0 slot
    for (int64_t t = 1; t < nthreads; t++)
        for (int64_t i = 0; i < M; i++) mean[i] += colsum[t * M + i];
    for (int64_t i = 0; i < M; i++) mean[i] /= (double)N;

    // ---- pass 2: center in place ----
    #pragma omp parallel for
    for (int64_t n = 0; n < N; n++) {
        double *row = data + n * M;
        #pragma omp simd
        for (int64_t i = 0; i < M; i++) row[i] -= mean[i];
    }
    free(colsum);

    // ---- pass 3: cov = D^T D upper triangle, mirrored ----
    const double inv = 1.0 / (float_n - 1.0);
    const int64_t L2 = 256, L1 = 64, L2K = 128;

    // block list: upper triangle of (L2-sized) blocks
    int64_t nb = 0;
    for (int64_t i0 = 0; i0 < M; i0 += L2)
        for (int64_t j0 = i0; j0 < M; j0 += L2) nb++;
    #pragma omp parallel for schedule(dynamic, 4)
    for (int64_t b = 0; b < nb; b++) {
        // find (i0, j0) for block b: solve row r: r(r+1)/2 <= count < (r+1)(r+2)/2
        int64_t r = (int64_t)((-1.0 + sqrt(1.0 + 8.0 * (double)b)) / 2.0);
        int64_t before = r * (r + 1) / 2;
        int64_t c = b - before;
        int64_t i0 = r * L2, j0 = (r + c) * L2;
        int64_t i1e = i0 + L2; if (i1e > M) i1e = M;
        int64_t j1e0 = j0 + L2; if (j1e0 > M) j1e0 = M;

        for (int64_t k0 = 0; k0 < N; k0 += L2K) {
            int64_t k1 = k0 + L2K; if (k1 > N) k1 = N;
            for (int64_t i1 = i0; i1 < i1e; i1 += L1) {
                int64_t i1x = i1 + L1; if (i1x > i1e) i1x = i1e;
                for (int64_t j1 = j0; j1 < j1e0; j1 += L1) {
                    int64_t j1x = j1 + L1; if (j1x > j1e0) j1x = j1e0;
                    // scratch accumulator for this L1 tile
                    double scr[L1 * L1];
                    memset(scr, 0, (size_t)L1 * L1 * sizeof(double));
                    for (int64_t mi = i1; mi < i1x; mi += MI) {
                        int64_t mie = mi + MI; if (mie > i1x) mie = i1x;
                        int64_t nrows = mie - mi;
                        for (int64_t nj = j1; nj < j1x; nj += JC) {
                            int64_t nje = nj + JC; if (nje > j1x) nje = j1x;
                            int64_t nchunks = (nje - nj) / SIMDD;
                            int64_t jtail = (nje - nj) % SIMDD;
                            // acc vectors: rows x chunks
                            v8d av[MI][NJ];
                            for (int64_t i = 0; i < nrows; i++)
                                for (int64_t c = 0; c < NJ; c++)
                                    av[i][c] = VZERO();
                            for (int64_t k = k0; k < k1; k++) {
                                const double *row = data + k * M;
                                double a[MI];
                                for (int64_t i = 0; i < nrows; i++) a[i] = row[mi + i];
                                for (int64_t c = 0; c < nchunks; c++) {
                                    v8d b = VLOAD(row + nj + c * SIMDD);
                                    for (int64_t i = 0; i < nrows; i++)
                                        av[i][c] = VFMA(VONE(a[i]), b, av[i][c]);
                                }
                                for (int64_t i = 0; i < nrows; i++)
                                    for (int64_t t = 0; t < jtail; t++)
                                        scr[(mi + i - i1) * L1 + (nj + (nchunks * SIMDD) + t - j1)] +=
                                            a[i] * row[nj + nchunks * SIMDD + t];
                            }
                            for (int64_t i = 0; i < nrows; i++)
                                for (int64_t c = 0; c < nchunks; c++)
                                    VSTORE(scr + (mi + i - i1) * L1 + (nj + c * SIMDD) - j1, av[i][c]);
                        }
                    }
                    // add scratch into cov (scaled)
                    for (int64_t i = 0; i < i1x - i1; i++) {
                        double *crow = cov + (i1 + i) * M + j1;
                        const double *srow = scr + i * L1;
                        #pragma omp simd
                        for (int64_t j = 0; j < j1x - j1; j++) crow[j] += srow[j] * inv;
                    }
                }
            }
        }
        // mirror the off-diagonal part of this L2 block onto the lower triangle
        if (i0 != j0) {
            for (int64_t i = 0; i < i1e - i0; i++)
                for (int64_t j = 0; j < j1e0 - j0; j++)
                    cov[(j0 + j) * M + (i0 + i)] = cov[(i0 + i) * M + (j0 + j)];
        }
    }
}
#else
#error "need AVX512"
#endif
