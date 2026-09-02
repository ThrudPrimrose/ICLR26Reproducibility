#include <stdint.h>
#include <omp.h>

static void tsvc_2_s119_large(double *restrict aa,
                              double *restrict bb,
                              int64_t n)
{
    constexpr int64_t TR = 64;
    constexpr int64_t TC = 128;
    const int64_t IR = (n + TR - 1) / TR;
    const int64_t JR = (n + TC - 1) / TC;

    #pragma omp parallel default(none) shared(aa, bb, n, IR, JR)
    {
        for (int64_t d = 0; d < IR + JR - 1; ++d) {
            const int64_t i0_start = (d < JR) ? 0 : d - (JR - 1);
            const int64_t i0_end   = (d < IR) ? d : IR - 1;
            #pragma omp for schedule(static)
            for (int64_t i0_t = i0_start; i0_t <= i0_end; ++i0_t) {
                const int64_t j0_t = d - i0_t;
                int64_t i0 = i0_t * TR;
                int64_t j0 = j0_t * TC;
                int64_t i_end = i0 + TR;
                int64_t j_end = j0 + TC;
                if (i_end > n) i_end = n;
                if (j_end > n) j_end = n;
                if (i0 < 1) i0 = 1;
                if (j0 < 1) j0 = 1;

                for (int64_t i = i0; i < i_end; ++i) {
                    double *restrict const arow  = aa + i * n;
                    double *restrict const arowm = aa + (i - 1) * n;
                    double *restrict const brow  = bb + i * n;
                    #pragma omp simd
                    for (int64_t j = j0; j < j_end; ++j) {
                        arow[j] = arowm[j - 1] + brow[j];
                    }
                }
            }
        }
    }
}

void tsvc_2_s119_fp64(double *restrict aa,
                      double *restrict bb,
                      int64_t n,
                      uint8_t *restrict ws,
                      int64_t ws_bytes)
{
    (void)ws;
    (void)ws_bytes;

    if (n < 2) return;

    if (n < 128) {
        for (int64_t i = 1; i < n; ++i) {
            double *restrict const arow  = aa + i * n;
            double *restrict const arowm = aa + (i - 1) * n;
            double *restrict const brow  = bb + i * n;
            #pragma omp simd
            for (int64_t j = 1; j < n; ++j) {
                arow[j] = arowm[j - 1] + brow[j];
            }
        }
        return;
    }

    tsvc_2_s119_large(aa, bb, n);
}
