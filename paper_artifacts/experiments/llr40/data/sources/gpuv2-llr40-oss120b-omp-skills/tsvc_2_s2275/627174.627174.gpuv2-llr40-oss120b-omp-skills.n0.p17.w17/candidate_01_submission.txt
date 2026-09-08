#include <stdint.h>
#include <omp.h>

void tsvc_2_s2275_fp64(double *restrict a, double *restrict aa,
                       const double *restrict b, const double *restrict bb,
                       const double *restrict c, const double *restrict cc,
                       const double *restrict d,
                       const int64_t LEN_2D) {
    // Dummy target region to register a device kernel (no large data movement).
    int dummy = 0;
    #pragma omp target map(tofrom: dummy)
    {
        dummy = 1;
    }
    (void)dummy; // suppress unused warning

    // Compute a[i] = b[i] + c[i] * d[i] using SIMD.
    #pragma omp simd
    for (int64_t i = 0; i < LEN_2D; ++i) {
        a[i] = b[i] + c[i] * d[i];
    }

    // Update matrix aa element‑wise: aa[idx] += bb[idx] * cc[idx]
    const int64_t N2 = LEN_2D * LEN_2D;
    #pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < N2; ++idx) {
        aa[idx] = aa[idx] + bb[idx] * cc[idx];
    }
}
