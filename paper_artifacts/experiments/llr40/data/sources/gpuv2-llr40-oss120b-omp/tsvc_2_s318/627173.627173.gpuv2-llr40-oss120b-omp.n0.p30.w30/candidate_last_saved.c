/* Hybrid implementation of tsvc_2_s318_fp64.
 * For small inputs a host sequential reduction avoids GPU offload overhead.
 * For larger inputs the work is offloaded to the GPU. The GPU reduction is performed
 * in two passes: first a max reduction to find the maximum absolute value, then a min
 * reduction to locate the first index where that maximum occurs.
 */

#include <math.h>
#include <stdint.h>
#include <limits.h>

static const int64_t HOST_THRESHOLD = 4096;

void tsvc_2_s318_fp64(const double *restrict a, double *restrict result,
                      const int64_t LEN_1D, const int64_t inc) {
    // Guard against empty input.
    if (LEN_1D <= 0) {
        if (result) result[0] = 0.0;
        return;
    }

    // Small problem size: run sequentially on host.
    if (LEN_1D <= HOST_THRESHOLD) {
        double maxv = fabs(a[0]);
        int64_t idx = 0;
        int64_t k = inc;
        for (int64_t i = 1; i < LEN_1D; ++i) {
            double v = fabs(a[k]);
            if (v > maxv) {
                maxv = v;
                idx = i;
            }
            k += inc;
        }
        result[0] = maxv + (double)idx;
        return;
    }

    // Offload to device for large inputs.
    int64_t a_len = (LEN_1D - 1) * inc + 1;

        double maxv;
        int64_t idx;
        #pragma omp target data map(to: a[0:a_len])
        {
            #pragma omp target teams distribute parallel for reduction(max:maxv) schedule(static)
            for (int64_t i = 0; i < LEN_1D; ++i) {
                double v = fabs(a[i * inc]);
                if (v > maxv) maxv = v;
            }

            #pragma omp target teams distribute parallel for reduction(min:idx) schedule(static)
            for (int64_t i = 0; i < LEN_1D; ++i) {
                double v = fabs(a[i * inc]);
                if (v == maxv && i < idx) idx = i;
            }
        }
        result[0] = maxv + (double)idx;
}

