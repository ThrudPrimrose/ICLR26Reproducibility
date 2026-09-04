#include <stdint.h>
#include <stdio.h>
#include <omp.h>

void versioned_distance_update_fp64(double *restrict a, const double *restrict b,
                                    const double *restrict c, const int64_t K,
                                    const int64_t LEN_1D) {
    (void)K;
    double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        a[i] = 0.5 * a[i] + b[i] * c[i];
    }
    double t1 = omp_get_wtime();
    fprintf(stdout, "BW_TEST: %.4f s for %.1f GB -> %.1f GB/s (serial 4-array pass)\n",
            t1 - t0, 4.0 * 8.0 * (double)LEN_1D / 1e9, 4.0 * 8.0 * (double)LEN_1D / 1e9 / (t1 - t0));
    fflush(stdout);
}
