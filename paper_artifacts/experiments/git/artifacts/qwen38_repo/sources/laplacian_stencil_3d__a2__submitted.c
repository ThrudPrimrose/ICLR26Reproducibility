// hpcagent_bench -- hand-optimized high-order 3-D Laplacian stencil (DFT kinetic operator).
//
// Computes, for a batch of k wavefunctions psi on an N^3 periodic grid:
//     lap  = inv_h2 * (Dx2 + Dy2 + Dz2) psi     (8th-order central 2nd derivative per axis)
//     ekin[j] = -0.5 * sum_{x,y,z} psi[x,y,z,j] * lap[x,y,z,j]
//
// Layout: psi[((x*N + y)*N + z)*k + kk]  (kk fastest, x slowest).
//
// Strategy:
//   * Single fused pass (no temporaries): the 25-point stencil is evaluated once per
//     element and the kinetic-energy accumulation is fused into the same pass.
//   * OpenMP over the outermost axis (x) with a static schedule.
//   * Explicit AVX-512 vectorisation over the fastest axis (kk, 8 doubles per 64-byte
//     vector).  The 24 neighbour loads are independent, so the hardware overlaps them
//     (latency hiding) and the number of memory-port operations per element drops by
//     8x versus a scalar inner loop.  A masked tail handles k % 8 != 0.
//   * The per-thread kinetic-energy accumulators live in 8 zmm registers (one per
//     8-kk group) and are reduced with atomics at the end.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>

// 8th-order central finite-difference coefficients of d^2/dx^2 (R = 4).
#define C0 (-205.0 / 72.0)
#define W1 (8.0 / 5.0)
#define W2 (-1.0 / 5.0)
#define W3 (8.0 / 315.0)
#define W4 (-1.0 / 560.0)
#define C_CENTER (3.0 * C0)

void laplacian_stencil_3d_fp64(double *restrict ekin, double *restrict lap,
                               const double *restrict psi, const int64_t N,
                               const double inv_h2, const int64_t k) {
    const int64_t N2 = N * N;
    const int64_t N2k = N2 * k;
    const int64_t Nk = N * k;

    // Precompute wrapped linear offsets for the stencil halo (m in -4..4).
    int64_t *offx = (int64_t *)malloc((size_t)(N * 9) * sizeof(int64_t));
    int64_t *offy = (int64_t *)malloc((size_t)(N * 9) * sizeof(int64_t));
    int64_t *offz = (int64_t *)malloc((size_t)(N * 9) * sizeof(int64_t));
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t m = -4; m <= 4; ++m) {
            int64_t j = i + m;
            if (j < 0) j += N; else if (j >= N) j -= N;
            offx[i * 9 + (m + 4)] = j * N2k;
            offy[i * 9 + (m + 4)] = j * Nk;
            offz[i * 9 + (m + 4)] = j * k;
        }
    }

    for (int64_t kk = 0; kk < k; ++kk) ekin[kk] = 0.0;

    const __m512d vc0 = _mm512_set1_pd(C_CENTER);
    const __m512d vw1 = _mm512_set1_pd(W1);
    const __m512d vw2 = _mm512_set1_pd(W2);
    const __m512d vw3 = _mm512_set1_pd(W3);
    const __m512d vw4 = _mm512_set1_pd(W4);
    const __m512d vih = _mm512_set1_pd(inv_h2);

    const int64_t nfull = k / 8;      // number of full 8-wide kk groups
    const int64_t ktail = nfull * 8;  // start index of the (masked) tail group
    const int64_t krow = ((k + 7) / 8) * 8;  // row width padded to a full vector (so the
    // tail's 8-wide store never spills into the neighbouring thread's row)
    const int nthreads = omp_get_max_threads();
    // Per-thread kinetic-energy buffers, reduced serially in a fixed order below so
    // the result is bit-reproducible run-to-run (no order-dependent FP atomics).
    double *thread_ekin = (double *)malloc((size_t)nthreads * krow * sizeof(double));
    memset(thread_ekin, 0, (size_t)nthreads * krow * sizeof(double));

#pragma omp parallel
    {
        __m512d me[8];
        for (int64_t g = 0; g < 8; ++g) me[g] = _mm512_setzero_pd();

#pragma omp for schedule(static)
        for (int64_t x = 0; x < N; ++x) {
            const int64_t *ox = offx + x * 9;
            const int64_t ox4 = ox[4];
            for (int64_t y = 0; y < N; ++y) {
                const int64_t *oy = offy + y * 9;
                const int64_t oy4 = oy[4];
                for (int64_t z = 0; z < N; ++z) {
                    const int64_t *oz = offz + z * 9;
                    const int64_t oz4 = oz[4];

                    const int64_t b_c   = ox4 + oy4 + oz4;
                    const int64_t b_xm1 = ox[3] + oy4 + oz4, b_xp1 = ox[5] + oy4 + oz4;
                    const int64_t b_xm2 = ox[2] + oy4 + oz4, b_xp2 = ox[6] + oy4 + oz4;
                    const int64_t b_xm3 = ox[1] + oy4 + oz4, b_xp3 = ox[7] + oy4 + oz4;
                    const int64_t b_xm4 = ox[0] + oy4 + oz4, b_xp4 = ox[8] + oy4 + oz4;
                    const int64_t b_ym1 = ox4 + oy[3] + oz4, b_yp1 = ox4 + oy[5] + oz4;
                    const int64_t b_ym2 = ox4 + oy[2] + oz4, b_yp2 = ox4 + oy[6] + oz4;
                    const int64_t b_ym3 = ox4 + oy[1] + oz4, b_yp3 = ox4 + oy[7] + oz4;
                    const int64_t b_ym4 = ox4 + oy[0] + oz4, b_yp4 = ox4 + oy[8] + oz4;
                    const int64_t b_zm1 = ox4 + oy4 + oz[3], b_zp1 = ox4 + oy4 + oz[5];
                    const int64_t b_zm2 = ox4 + oy4 + oz[2], b_zp2 = ox4 + oy4 + oz[6];
                    const int64_t b_zm3 = ox4 + oy4 + oz[1], b_zp3 = ox4 + oy4 + oz[7];
                    const int64_t b_zm4 = ox4 + oy4 + oz[0], b_zp4 = ox4 + oy4 + oz[8];

                    // Full 8-wide kk groups.
                    for (int64_t o = 0; o < ktail; o += 8) {
                        const __m512d pc  = _mm512_loadu_pd(psi + b_c   + o);
                        __m512d acc = _mm512_mul_pd(pc, vc0);
                        __m512d s = _mm512_add_pd(_mm512_loadu_pd(psi + b_xm1 + o), _mm512_loadu_pd(psi + b_xp1 + o));
                        acc = _mm512_fmadd_pd(vw1, s, acc);
                        s = _mm512_add_pd(_mm512_loadu_pd(psi + b_xm2 + o), _mm512_loadu_pd(psi + b_xp2 + o));
                        acc = _mm512_fmadd_pd(vw2, s, acc);
                        s = _mm512_add_pd(_mm512_loadu_pd(psi + b_xm3 + o), _mm512_loadu_pd(psi + b_xp3 + o));
                        acc = _mm512_fmadd_pd(vw3, s, acc);
                        s = _mm512_add_pd(_mm512_loadu_pd(psi + b_xm4 + o), _mm512_loadu_pd(psi + b_xp4 + o));
                        acc = _mm512_fmadd_pd(vw4, s, acc);
                        s = _mm512_add_pd(_mm512_loadu_pd(psi + b_ym1 + o), _mm512_loadu_pd(psi + b_yp1 + o));
                        acc = _mm512_fmadd_pd(vw1, s, acc);
                        s = _mm512_add_pd(_mm512_loadu_pd(psi + b_ym2 + o), _mm512_loadu_pd(psi + b_yp2 + o));
                        acc = _mm512_fmadd_pd(vw2, s, acc);
                        s = _mm512_add_pd(_mm512_loadu_pd(psi + b_ym3 + o), _mm512_loadu_pd(psi + b_yp3 + o));
                        acc = _mm512_fmadd_pd(vw3, s, acc);
                        s = _mm512_add_pd(_mm512_loadu_pd(psi + b_ym4 + o), _mm512_loadu_pd(psi + b_yp4 + o));
                        acc = _mm512_fmadd_pd(vw4, s, acc);
                        s = _mm512_add_pd(_mm512_loadu_pd(psi + b_zm1 + o), _mm512_loadu_pd(psi + b_zp1 + o));
                        acc = _mm512_fmadd_pd(vw1, s, acc);
                        s = _mm512_add_pd(_mm512_loadu_pd(psi + b_zm2 + o), _mm512_loadu_pd(psi + b_zp2 + o));
                        acc = _mm512_fmadd_pd(vw2, s, acc);
                        s = _mm512_add_pd(_mm512_loadu_pd(psi + b_zm3 + o), _mm512_loadu_pd(psi + b_zp3 + o));
                        acc = _mm512_fmadd_pd(vw3, s, acc);
                        s = _mm512_add_pd(_mm512_loadu_pd(psi + b_zm4 + o), _mm512_loadu_pd(psi + b_zp4 + o));
                        acc = _mm512_fmadd_pd(vw4, s, acc);
                        const __m512d lv = _mm512_mul_pd(acc, vih);
                        _mm512_storeu_pd(lap + b_c + o, lv);
                        me[o / 8] = _mm512_fmadd_pd(pc, lv, me[o / 8]);
                    }
                    // Masked tail (k % 8 != 0).
                    if (ktail < k) {
                        const int n = (int)(k - ktail);
                        const __mmask8 mask = (__mmask8)((1u << n) - 1u);
                        const __m512d pc = _mm512_maskz_loadu_pd(mask, psi + b_c + ktail);
                        __m512d acc = _mm512_mul_pd(pc, vc0);
                        __m512d s = _mm512_add_pd(_mm512_maskz_loadu_pd(mask, psi + b_xm1 + ktail), _mm512_maskz_loadu_pd(mask, psi + b_xp1 + ktail));
                        acc = _mm512_fmadd_pd(vw1, s, acc);
                        s = _mm512_add_pd(_mm512_maskz_loadu_pd(mask, psi + b_xm2 + ktail), _mm512_maskz_loadu_pd(mask, psi + b_xp2 + ktail));
                        acc = _mm512_fmadd_pd(vw2, s, acc);
                        s = _mm512_add_pd(_mm512_maskz_loadu_pd(mask, psi + b_xm3 + ktail), _mm512_maskz_loadu_pd(mask, psi + b_xp3 + ktail));
                        acc = _mm512_fmadd_pd(vw3, s, acc);
                        s = _mm512_add_pd(_mm512_maskz_loadu_pd(mask, psi + b_xm4 + ktail), _mm512_maskz_loadu_pd(mask, psi + b_xp4 + ktail));
                        acc = _mm512_fmadd_pd(vw4, s, acc);
                        s = _mm512_add_pd(_mm512_maskz_loadu_pd(mask, psi + b_ym1 + ktail), _mm512_maskz_loadu_pd(mask, psi + b_yp1 + ktail));
                        acc = _mm512_fmadd_pd(vw1, s, acc);
                        s = _mm512_add_pd(_mm512_maskz_loadu_pd(mask, psi + b_ym2 + ktail), _mm512_maskz_loadu_pd(mask, psi + b_yp2 + ktail));
                        acc = _mm512_fmadd_pd(vw2, s, acc);
                        s = _mm512_add_pd(_mm512_maskz_loadu_pd(mask, psi + b_ym3 + ktail), _mm512_maskz_loadu_pd(mask, psi + b_yp3 + ktail));
                        acc = _mm512_fmadd_pd(vw3, s, acc);
                        s = _mm512_add_pd(_mm512_maskz_loadu_pd(mask, psi + b_ym4 + ktail), _mm512_maskz_loadu_pd(mask, psi + b_yp4 + ktail));
                        acc = _mm512_fmadd_pd(vw4, s, acc);
                        s = _mm512_add_pd(_mm512_maskz_loadu_pd(mask, psi + b_zm1 + ktail), _mm512_maskz_loadu_pd(mask, psi + b_zp1 + ktail));
                        acc = _mm512_fmadd_pd(vw1, s, acc);
                        s = _mm512_add_pd(_mm512_maskz_loadu_pd(mask, psi + b_zm2 + ktail), _mm512_maskz_loadu_pd(mask, psi + b_zp2 + ktail));
                        acc = _mm512_fmadd_pd(vw2, s, acc);
                        s = _mm512_add_pd(_mm512_maskz_loadu_pd(mask, psi + b_zm3 + ktail), _mm512_maskz_loadu_pd(mask, psi + b_zp3 + ktail));
                        acc = _mm512_fmadd_pd(vw3, s, acc);
                        s = _mm512_add_pd(_mm512_maskz_loadu_pd(mask, psi + b_zm4 + ktail), _mm512_maskz_loadu_pd(mask, psi + b_zp4 + ktail));
                        acc = _mm512_fmadd_pd(vw4, s, acc);
                        const __m512d lv = _mm512_mul_pd(acc, vih);
                        _mm512_mask_storeu_pd(lap + b_c + ktail, mask, lv);
                        me[nfull] = _mm512_fmadd_pd(pc, lv, me[nfull]);
                    }
                }
            }
        }

        // Spill this thread's kinetic-energy accumulators into its buffer row.
        double *tb = thread_ekin + (size_t)omp_get_thread_num() * krow;
        for (int64_t g = 0; g < nfull; ++g) _mm512_storeu_pd(tb + g * 8, me[g]);
        if (ktail < k) _mm512_storeu_pd(tb + ktail, me[nfull]);
    }

    // Deterministic serial reduction (fixed thread order) + the -0.5 scale.
    for (int64_t kk = 0; kk < k; ++kk) {
        double s = 0.0;
        for (int tid = 0; tid < nthreads; ++tid) s += thread_ekin[(size_t)tid * krow + kk];
        ekin[kk] = -0.5 * s;
    }
    free(thread_ekin);

    free(offx);
    free(offy);
    free(offz);
}
