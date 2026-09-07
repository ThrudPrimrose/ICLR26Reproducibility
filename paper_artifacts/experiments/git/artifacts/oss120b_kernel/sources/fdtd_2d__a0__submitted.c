/* Optimized FDtd 2D kernel with adaptive serial/parallel execution and SIMD. */
#include <stddef.h>
#include <stdint.h>
#include <omp.h>

void fdtd_2d_fp64(double *restrict ex, double *restrict ey, const double *restrict fict,
                  double *restrict hz, int64_t NX, int64_t NY, int64_t TMAX,
                  double ex_courant, double ey_courant, double hz_courant) {
    const double ex_c = ex_courant;
    const double ey_c = ey_courant;
    const double hz_c = hz_courant;

    // If the problem size is tiny, run the simple serial version to avoid parallel overhead.
    if (NX * NY <= 1024) {
        for (int64_t t = 0; t < TMAX; ++t) {
            // ey[0, :]
            for (int64_t si1 = 0; si1 < NY; ++si1) {
                ey[si1] = fict[t];
            }
            // ey[1:, :]
            for (int64_t si0 = 1; si0 < NX; ++si0) {
                for (int64_t si1 = 0; si1 < NY; ++si1) {
                    const int64_t idx = si0 * NY + si1;
                    const int64_t idx_up = (si0 - 1) * NY + si1;
                    ey[idx] -= ey_c * (hz[idx] - hz[idx_up]);
                }
            }
            // ex[:, 1:]
            for (int64_t si0 = 0; si0 < NX; ++si0) {
                for (int64_t si1 = 1; si1 < NY; ++si1) {
                    const int64_t idx = si0 * NY + si1;
                    const int64_t idx_left = si0 * NY + (si1 - 1);
                    ex[idx] -= ex_c * (hz[idx] - hz[idx_left]);
                }
            }
            // hz[:NX-1, :NY-1]
            for (int64_t si0 = 0; si0 < NX - 1; ++si0) {
                for (int64_t si1 = 0; si1 < NY - 1; ++si1) {
                    const int64_t idx = si0 * NY + si1;
                    const int64_t idx_ex_n = si0 * NY + (si1 + 1);
                    const int64_t idx_ey_n = (si0 + 1) * NY + si1;
                    hz[idx] -= hz_c * ((ex[idx_ex_n] - ex[idx]) + (ey[idx_ey_n] - ey[idx]));
                }
            }
        }
        return;
    }

    // Parallel version for larger problems.

        // Align pointers to improve SIMD performance for large datasets
    if (NX * NY >= 65536) {
        ex = (double *)__builtin_assume_aligned(ex, 64);
        ey = (double *)__builtin_assume_aligned(ey, 64);
        hz = (double *)__builtin_assume_aligned(hz, 64);
    }
    #pragma omp parallel
    {
        for (int64_t t = 0; t < TMAX; ++t) {
            // Update ey first row
            #pragma omp for nowait
            for (int64_t si0 = 0; si0 < NX; ++si0) {
                double *restrict cur_ey = ey + si0 * NY;
                if (si0 == 0) {
                    #pragma omp simd
                    for (int64_t si1 = 0; si1 < NY; ++si1) {
                        cur_ey[si1] = fict[t];
                    }
                } else {
                    const double *restrict cur_hz = hz + si0 * NY;
                    const double *restrict prev_hz = hz + (si0 - 1) * NY;
                    #pragma omp simd
                    for (int64_t si1 = 0; si1 < NY; ++si1) {
                        cur_ey[si1] -= ey_c * (cur_hz[si1] - prev_hz[si1]);
                    }
                }
            }
            // Update ex rows (all rows)
            #pragma omp for nowait
            for (int64_t si0 = 0; si0 < NX; ++si0) {
                double *restrict ex_row = ex + si0 * NY;
                const double *restrict hz_row = hz + si0 * NY;
                #pragma omp simd
                for (int64_t si1 = 1; si1 < NY; ++si1) {
                    ex_row[si1] -= ex_c * (hz_row[si1] - hz_row[si1 - 1]);
                }
            }
            // Synchronize before hz update
            #pragma omp barrier
            // Update hz interior (rows 0..NX-2, cols 0..NY-2)
            #pragma omp for
            for (int64_t si0 = 0; si0 < NX - 1; ++si0) {
                double *restrict hz_row = hz + si0 * NY;
                const double *restrict ex_row = ex + si0 * NY;
                const double *restrict ey_row = ey + si0 * NY;
                const double *restrict ey_next = ey + (si0 + 1) * NY;
                #pragma omp simd
                for (int64_t si1 = 0; si1 < NY - 1; ++si1) {
                    hz_row[si1] -= hz_c * ((ex_row[si1 + 1] - ex_row[si1]) + (ey_next[si1] - ey_row[si1]));
                }
            }
        }
    }
}
