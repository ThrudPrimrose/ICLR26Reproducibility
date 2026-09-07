#include <stdint.h>
#include <stdlib.h>

static void single_step(const double *restrict src, double *restrict dst,
                        int64_t N, double alpha)
{
    const double alpha6m1 = 1.0 - 6.0 * alpha;
    const int64_t N2 = N * N;

    #pragma omp parallel for collapse(3) schedule(static)
    for (int64_t i = 1; i < N - 1; ++i) {
        for (int64_t j = 1; j < N - 1; ++j) {
            for (int64_t k = 1; k < N - 1; ++k) {
                const double c = src[i * N2 + j * N + k];
                const double s = src[(i + 1) * N2 + j * N + k]
                                 + src[(i - 1) * N2 + j * N + k]
                                 + src[i * N2 + (j + 1) * N + k]
                                 + src[i * N2 + (j - 1) * N + k]
                                 + src[i * N2 + j * N + (k + 1)]
                                 + src[i * N2 + j * N + (k - 1)];
                dst[i * N2 + j * N + k] = alpha * s + alpha6m1 * c;
            }
        }
    }
}

static inline int64_t clamp(int64_t v, int64_t lo, int64_t hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static void temporal_pair(const double *restrict src, double *restrict dst,
                          int64_t N, double alpha, int64_t B)
{
    const double alpha6m1 = 1.0 - 6.0 * alpha;
    const int64_t N2 = N * N;
    const int64_t Bl4 = B + 4;
    const int64_t Bl2 = B + 2;

    #pragma omp parallel
    {
        double *abuf = (double *)malloc((size_t)(Bl4 * Bl4 * Bl4) * sizeof(double));
        double *bbuf = (double *)malloc((size_t)(Bl2 * Bl2 * Bl2) * sizeof(double));

        #pragma omp for collapse(3) schedule(static)
        for (int64_t bx0 = 1; bx0 < N - 1; bx0 += B) {
            for (int64_t by0 = 1; by0 < N - 1; by0 += B) {
                for (int64_t bz0 = 1; bz0 < N - 1; bz0 += B) {
                    const int64_t x1 = bx0;
                    const int64_t x2 = bx0 + B - 1 < N - 2 ? bx0 + B - 1 : N - 2;
                    const int64_t y1 = by0;
                    const int64_t y2 = by0 + B - 1 < N - 2 ? by0 + B - 1 : N - 2;
                    const int64_t z1 = bz0;
                    const int64_t z2 = bz0 + B - 1 < N - 2 ? bz0 + B - 1 : N - 2;

                    const int64_t ex = x2 - x1 + 1;
                    const int64_t ey = y2 - y1 + 1;
                    const int64_t ez = z2 - z1 + 1;

                    const int64_t ax = ex + 4;
                    const int64_t ay = ey + 4;
                    const int64_t az = ez + 4;
                    const int64_t bx_ = ex + 2;
                    const int64_t by_ = ey + 2;
                    const int64_t bz_ = ez + 2;

                    for (int64_t i = 0; i < ax; ++i) {
                        const int64_t gi = clamp(x1 - 2 + i, 0, N - 1);
                        for (int64_t j = 0; j < ay; ++j) {
                            const int64_t gj = clamp(y1 - 2 + j, 0, N - 1);
                            for (int64_t k = 0; k < az; ++k) {
                                const int64_t gk = clamp(z1 - 2 + k, 0, N - 1);
                                abuf[(i * ay + j) * az + k] = src[gi * N2 + gj * N + gk];
                            }
                        }
                    }

                    for (int64_t i = 0; i < bx_; ++i) {
                        const int64_t gx = bx0 - 1 + i;
                        for (int64_t j = 0; j < by_; ++j) {
                            const int64_t gy = by0 - 1 + j;
                            for (int64_t k = 0; k < bz_; ++k) {
                                const int64_t gz = bz0 - 1 + k;
                                if (gx == 0 || gx == N - 1 ||
                                    gy == 0 || gy == N - 1 ||
                                    gz == 0 || gz == N - 1) {
                                    bbuf[(i * by_ + j) * bz_ + k] = src[gx * N2 + gy * N + gz];
                                } else {
                                    const double c = abuf[((i + 1) * ay + (j + 1)) * az + (k + 1)];
                                    const double s = abuf[((i + 2) * ay + (j + 1)) * az + (k + 1)]
                                                     + abuf[((i + 0) * ay + (j + 1)) * az + (k + 1)]
                                                     + abuf[((i + 1) * ay + (j + 2)) * az + (k + 1)]
                                                     + abuf[((i + 1) * ay + (j + 0)) * az + (k + 1)]
                                                     + abuf[((i + 1) * ay + (j + 1)) * az + (k + 2)]
                                                     + abuf[((i + 1) * ay + (j + 1)) * az + (k + 0)];
                                    bbuf[(i * by_ + j) * bz_ + k] = alpha * s + alpha6m1 * c;
                                }
                            }
                        }
                    }

                    for (int64_t i = 0; i < ex; ++i) {
                        const int64_t gi = x1 + i;
                        for (int64_t j = 0; j < ey; ++j) {
                            const int64_t gj = y1 + j;
                            for (int64_t k = 0; k < ez; ++k) {
                                const int64_t gk = z1 + k;
                                const double c = bbuf[((i + 1) * by_ + (j + 1)) * bz_ + (k + 1)];
                                const double s = bbuf[((i + 2) * by_ + (j + 1)) * bz_ + (k + 1)]
                                                 + bbuf[((i + 0) * by_ + (j + 1)) * bz_ + (k + 1)]
                                                 + bbuf[((i + 1) * by_ + (j + 2)) * bz_ + (k + 1)]
                                                 + bbuf[((i + 1) * by_ + (j + 0)) * bz_ + (k + 1)]
                                                 + bbuf[((i + 1) * by_ + (j + 1)) * bz_ + (k + 2)]
                                                 + bbuf[((i + 1) * by_ + (j + 1)) * bz_ + (k + 0)];
                                dst[gi * N2 + gj * N + gk] = alpha * s + alpha6m1 * c;
                            }
                        }
                    }
                }
            }
        }

        free(abuf);
        free(bbuf);
    }
}

void heat_3d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS, double alpha) {
    const int64_t BS = 28;
    const int64_t pairs = TSTEPS / 2;

    for (int64_t p = 0; p < pairs; ++p) {
        const double *src = (p % 2 == 0) ? A : B;
        double *dst = (p % 2 == 0) ? B : A;
        temporal_pair(src, dst, N, alpha, BS);
    }

    if (TSTEPS % 2 != 0) {
        if (pairs % 2 == 0) {
            single_step(A, B, N, alpha);
            const int64_t NN = N * N * N;
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < NN; ++i) A[i] = B[i];
        } else {
            single_step(B, A, N, alpha);
        }
    } else {
        if (pairs % 2 != 0) {
            const int64_t NN = N * N * N;
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < NN; ++i) A[i] = B[i];
        }
    }
}
