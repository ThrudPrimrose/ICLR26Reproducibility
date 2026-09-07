#include <stdint.h>
#include <omp.h>

void edge_laplacian_fp64(double *restrict Lx,
                         const int64_t *restrict dst,
                         const int64_t *restrict src,
                         const double *restrict w,
                         const double *restrict x,
                         const int64_t E,
                         const int64_t N)
{
    #pragma omp parallel for
    for (int64_t i = 0; i < N; ++i) {
        Lx[i] = 0.0;
    }

    #pragma omp parallel for schedule(static)
    for (int64_t e = 0; e < E; ++e) {
        int64_t s = src[e];
        int64_t d = dst[e];
        double flux = w[e] * (x[s] - x[d]);
        #pragma omp atomic update
        Lx[s] += flux;
        #pragma omp atomic update
        Lx[d] -= flux;
    }
}
