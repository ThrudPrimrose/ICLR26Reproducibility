/* Optimized OpenMP implementation of fdtd_2d kernel in double precision.
 * Uses a single parallel region across time steps, vectorizes inner loops,
 * and adaptively selects the number of OpenMP threads based on problem size.
 */
#include <stddef.h>
#include <stdint.h>
#include <omp.h>

void fdtd_2d_fp64(double *restrict ex, double *restrict ey, const double *restrict fict,
                  double *restrict hz, int64_t NX, int64_t NY, int64_t TMAX,
                  double ex_courant, double ey_courant, double hz_courant) {
    const double ex_c = ex_courant;
    const double ey_c = ey_courant;
    const double hz_c = hz_courant;

    // Determine a reasonable thread count. For very small problems many threads hurt performance.
    // Use at most one thread per ~100k grid points, capped by the system's max threads.
    int64_t total_cells = NX * NY;
    int max_threads = omp_get_max_threads();
    int desired_threads = max_threads;
    int64_t per_thread_min_cells = 100000; // heuristic
    if (total_cells / per_thread_min_cells < max_threads) {
        desired_threads = (int)(total_cells / per_thread_min_cells);
        if (desired_threads < 1) desired_threads = 1;
    }
    omp_set_num_threads(desired_threads);

    #pragma omp parallel
    {
        for (int64_t t = 0; t < TMAX; ++t) {
            /* ey[0, :] = fict[t] */
            #pragma omp for schedule(static)
            for (int64_t si1 = 0; si1 < NY; ++si1) {
                ey[si1] = fict[t];
            }

            /* ey[1:, :] update */
            #pragma omp for schedule(static)
            for (int64_t si0 = 1; si0 < NX; ++si0) {
                const int64_t base = si0 * NY;
                const int64_t base_up = (si0 - 1) * NY;
                #pragma omp simd
                for (int64_t si1 = 0; si1 < NY; ++si1) {
                    ey[base + si1] -= ey_c * (hz[base + si1] - hz[base_up + si1]);
                }
            }

            /* ex[:, 1:] update */
            #pragma omp for schedule(static)
            for (int64_t si0 = 0; si0 < NX; ++si0) {
                const int64_t base = si0 * NY;
                #pragma omp simd
                for (int64_t si1 = 1; si1 < NY; ++si1) {
                    ex[base + si1] -= ex_c * (hz[base + si1] - hz[base + si1 - 1]);
                }
            }

            /* hz[:NX-1, :NY-1] update */
            #pragma omp for schedule(static)
            for (int64_t si0 = 0; si0 < NX - 1; ++si0) {
                const int64_t base = si0 * NY;
                const int64_t base_next = (si0 + 1) * NY;
                #pragma omp simd
                for (int64_t si1 = 0; si1 < NY - 1; ++si1) {
                    const int64_t idx = base + si1;
                    const int64_t idx_ex_n = base + si1 + 1;
                    const int64_t idx_ey_n = base_next + si1;
                    hz[idx] -= hz_c * ((ex[idx_ex_n] - ex[idx]) + (ey[idx_ey_n] - ey[idx]));
                }
            }
            // implicit barrier after each omp for ensures correct ordering.
        }
    }
}
