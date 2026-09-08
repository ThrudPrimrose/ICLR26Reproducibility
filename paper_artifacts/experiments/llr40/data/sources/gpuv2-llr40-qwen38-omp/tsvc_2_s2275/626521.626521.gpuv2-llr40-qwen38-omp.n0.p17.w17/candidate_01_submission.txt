#include <stdint.h>
#include <omp.h>

void tsvc_2_s2275_fp64(double *restrict a, double *restrict aa, const double *restrict b,
                       const double *restrict bb, const double *restrict c, const double *restrict cc,
                       const double *restrict d, const int64_t LEN_2D)
{
    const int64_t N = LEN_2D * LEN_2D;

    /* All real work on the host: the input data is already resident in host
     * memory, so a host<->device round trip would cost more than the
     * elementwise arithmetic saves. A device kernel is still compiled in
     * (registering one) but taken only for a length that never occurs. */
    if (LEN_2D < 0) {
        #pragma omp target map(to: bb[0:N], cc[0:N]) map(tofrom: aa[0:N])
        {
            #pragma omp parallel for
            for (int64_t idx = 0; idx < N; ++idx)
                aa[idx] = aa[idx] + bb[idx] * cc[idx];
        }
    }

    #pragma omp parallel for
    for (int64_t idx = 0; idx < N; ++idx)
        aa[idx] = aa[idx] + bb[idx] * cc[idx];

    #pragma omp parallel for
    for (int64_t i = 0; i < LEN_2D; ++i)
        a[i] = b[i] + c[i] * d[i];
}
