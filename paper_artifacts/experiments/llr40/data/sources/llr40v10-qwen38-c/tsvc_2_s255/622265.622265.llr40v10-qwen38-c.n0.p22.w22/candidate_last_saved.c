#include <stdint.h>

/* TSVC s255.  The two carry-around scalars x,y only ever hold b[i-1] and
 * b[i-2] (circularly), so the loop is really the 3-point stencil
 *     a[i] = (b[i] + b[i-1] + b[i-2]) * 0.333
 * with wrap for i=0,1.  There is NO true serial dependence, so the body is
 * fully SIMD- and thread-parallel.  For small LEN_1D the OpenMP team fork
 * (~tens of us) would dominate, so we run a serial (auto-vectorized) loop;
 * for large LEN_1D we parallelize. */

void tsvc_2_s255_fp64(double *restrict a, const double *restrict b,
                      const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;
    const double c = 0.333;

    a[0] = (b[0] + b[LEN_1D-1] + b[LEN_1D-2]) * c;
    if (LEN_1D > 1)
        a[1] = (b[1] + b[0] + b[LEN_1D-1]) * c;

    if (LEN_1D < 100000) {
        for (int64_t i = 2; i < LEN_1D; i++)
            a[i] = (b[i] + b[i-1] + b[i-2]) * c;
    } else {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 2; i < LEN_1D; i++)
            a[i] = (b[i] + b[i-1] + b[i-2]) * c;
    }
}
