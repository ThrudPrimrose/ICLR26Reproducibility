#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

/* Optimized version of tsvc_2_s252_fp64 for OpenMP offload.
 * The original recurrence a[i] = b[i]*c[i] + t (where t holds the previous product)
 * is transformed by first computing the element‑wise product into a temporary buffer
 * `tmp`, then forming a[i] = tmp[i] + tmp[i-1] (with a[0] = tmp[0]). This eliminates the
 * loop‑carried scalar dependency and enables vectorization. The work is placed inside
 * explicit OpenMP target regions so that the kernel runs on the GPU as required by the
 * offload arm.
 */

void tsvc_2_s252_fp64(double *restrict a,
                      const double *restrict b,
                      const double *restrict c,
                      const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;

    // Allocate a temporary buffer on the host; it will be allocated on the device via map(alloc).
    double *tmp = (double *)malloc((size_t)LEN_1D * sizeof(double));
    if (!tmp) {
        // Fallback to a pure host implementation if allocation fails.
        double t = 0.0;
        for (int64_t i = 0; i < LEN_1D; ++i) {
            double s = b[i] * c[i];
            a[i] = s + t;
            t = s;
        }
        return;
    }

    // Offload the computation to the device.
    #pragma omp target data map(to: b[0:LEN_1D], c[0:LEN_1D]) \
                            map(tofrom: a[0:LEN_1D]) map(alloc: tmp[0:LEN_1D])
    {
        // First pass: element‑wise product.
        #pragma omp target teams distribute parallel for
        for (int64_t i = 0; i < LEN_1D; ++i) {
            tmp[i] = b[i] * c[i];
        }

        // Second pass: compute final result using the shifted sum.
        #pragma omp target teams distribute parallel for
        for (int64_t i = 0; i < LEN_1D; ++i) {
            if (i == 0) {
                a[i] = tmp[i];
            } else {
                a[i] = tmp[i] + tmp[i - 1];
            }
        }
    }

    free(tmp);
}
