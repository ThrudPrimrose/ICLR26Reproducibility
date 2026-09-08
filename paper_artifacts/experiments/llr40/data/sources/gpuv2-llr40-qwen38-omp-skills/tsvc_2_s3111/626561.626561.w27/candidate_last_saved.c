#include <stdint.h>
#include <string.h>
#include <omp.h>

/* Sum of positive elements of a[0..LEN_1D) into b[0].
 *
 * libomptarget on this ROCm build never finalizes `reduction(+:)` on the
 * device (result scalar is not copied back, stays 0), so partials are
 * collected with atomic updates into a small per-team buffer. */

#define TP 1024

void tsvc_2_s3111_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
    if (LEN_1D <= 0) {
        b[0] = 0.0;
        return;
    }

    if (LEN_1D < 524288) {
        /* small input: host path, the GPU launch + map round trip does not pay */
        double s = 0.0;
        int64_t i = 0;
        const int64_t n8 = (LEN_1D & ~7LL) / 8;
        for (int64_t k = 0; k < n8; ++k) {
            const double *p = a + k * 8;
            s += (p[0] > 0.0) ? p[0] : 0.0;
            s += (p[1] > 0.0) ? p[1] : 0.0;
            s += (p[2] > 0.0) ? p[2] : 0.0;
            s += (p[3] > 0.0) ? p[3] : 0.0;
            s += (p[4] > 0.0) ? p[4] : 0.0;
            s += (p[5] > 0.0) ? p[5] : 0.0;
            s += (p[6] > 0.0) ? p[6] : 0.0;
            s += (p[7] > 0.0) ? p[7] : 0.0;
        }
        i = n8 * 8;
        for (; i < LEN_1D; ++i) s += (a[i] > 0.0) ? a[i] : 0.0;
        b[0] = s;
        return;
    }

    double pt[TP];
    memset(pt, 0, sizeof(pt));

    #pragma omp target map(to: a[0:LEN_1D]) map(tofrom: pt[0:TP])
        #pragma omp teams distribute parallel for
        for (int64_t i = 0; i < LEN_1D; ++i) {
            int64_t t = omp_get_team_num() % TP;
            #pragma omp atomic update
            pt[t] += (a[i] > 0.0) ? a[i] : 0.0;
        }

    double s = 0.0;
    for (int64_t i = 0; i < TP; ++i) s += pt[i];
    b[0] = s;
}
