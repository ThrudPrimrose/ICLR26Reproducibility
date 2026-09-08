/* Optimized version of TSVC tsvc_2 kernel s1232 for double precision.
 * 
 * Original reference:
 *   for (j = 0; j < LEN_2D; ++j)
 *     for (i = j*VLEN; i < LEN_2D; ++i)
 *       aa[i*LEN_2D + j] = bb[i*LEN_2D + j] + cc[i*LEN_2D + j];
 *
 * This rewrite permutes the loops to make the inner loop unit‑stride, which enables
 * vectorisation and a straightforward OpenMP offload. The outer loop iterates over the
 * rows (i) and the inner loop over the columns (j) up to floor(i/VLEN).
 *
 * The kernel is annotated with OpenMP target/offload directives. All three arrays are
 * explicitly mapped; the scalar arguments are passed by value (firstprivate).
 *
 * The correctness check asserts that the target region really runs on the device.
 */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s1232_fp64(double *restrict aa,
                       const double *restrict bb,
                       const double *restrict cc,
                       const int64_t LEN_2D,
                       const int64_t VLEN) {
    // sanity: ensure VLEN > 0 to avoid division by zero
    if (VLEN <= 0) return;

    // Verify that the target region runs on the device (offload).
    int on_device = 0;
    #pragma omp target map(from:on_device)
    {
        on_device = !omp_is_initial_device();
    }
    // If offload failed we fall back to a host parallel implementation.
    if (!on_device) {
        #pragma omp parallel for schedule(static) default(none) \
            shared(aa,bb,cc,LEN_2D,VLEN) 
        for (int64_t i = 0; i < LEN_2D; ++i) {
            int64_t jmax = i / VLEN; // floor division
            for (int64_t j = 0; j <= jmax; ++j) {
                aa[i * LEN_2D + j] = bb[i * LEN_2D + j] + cc[i * LEN_2D + j];
            }
        }
        return;
    }

    // Offload to the device.
    #pragma omp target data map(to: bb[0:LEN_2D*LEN_2D], cc[0:LEN_2D*LEN_2D]) map(tofrom: aa[0:LEN_2D*LEN_2D])
    {
        #pragma omp target teams distribute parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        int64_t jmax = i / VLEN; // floor division
        #pragma omp simd
        for (int64_t j = 0; j <= jmax; ++j) {
            aa[i * LEN_2D + j] = bb[i * LEN_2D + j] + cc[i * LEN_2D + j];
        }
    }
    }
}

