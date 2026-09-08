/* Optimized version of tsvc_2_s311 kernel: sum of array into a scalar output.
   Uses OpenMP target offload with reduction and SIMD for vectorization.
   The function signature matches the reference exactly. */
#include <stdint.h>
#include <omp.h>

void tsvc_2_s311_fp64(const double *restrict a, double *restrict sum_out, const int64_t LEN_1D) {
    // Check that we are executing on the device (optional sanity check).
    int on_device = 0;
    #pragma omp target map(from:on_device)
    {
        on_device = !omp_is_initial_device();
    }
    // Compute the sum on the device.
    double sum = 0.0;
    #pragma omp target teams distribute parallel for simd map(to:a[0:LEN_1D]) reduction(+:sum) schedule(static)
    for (int64_t i = 0; i < LEN_1D; i++) {
        sum += a[i];
    }
    // Write the result back to the output array.
    sum_out[0] = sum;
}
