#include <stdint.h>
#include <omp.h>

void edge_laplacian_fp64(const int32_t *restrict src,
                         const int32_t *restrict dst,
                         const double *restrict w,
                         const double *restrict x,
                         double *restrict Lx,
                         int64_t N,
                         int64_t E) {
    for (int64_t i = 0; i < N; ++i) {
        Lx[i] = 0.0;
    }

    for (int64_t e = 0; e < E; ++e) {
        double flux = w[e] * (x[src[e]] - x[dst[e]]);
        Lx[src[e]] += flux;
        Lx[dst[e]] -= flux;
    }
}
