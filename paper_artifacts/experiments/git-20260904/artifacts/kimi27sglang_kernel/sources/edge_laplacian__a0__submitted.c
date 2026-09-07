#include <stdint.h>
#include <stdio.h>

void edge_laplacian_fp64(int64_t *restrict src,
                         int64_t *restrict dst,
                         double *restrict w,
                         double *restrict x,
                         double *restrict Lx,
                         int64_t N,
                         int64_t E)
{
    fprintf(stderr, "DEBUG N=%lld E=%lld src0=%lld dst0=%lld x0=%g w0=%g\n",
            (long long)N, (long long)E,
            (E>0?(long long)src[0]:-1),
            (E>0?(long long)dst[0]:-1),
            (N>0?x[0]:0.0),
            (E>0?w[0]:0.0));
    fflush(stderr);
    for (int64_t i = 0; i < N; ++i) {
        Lx[i] = 0.0;
    }

    for (int64_t e = 0; e < E; ++e) {
        int64_t s = src[e];
        int64_t d = dst[e];
        double flux = w[e] * (x[s] - x[d]);
        Lx[s] += flux;
        Lx[d] -= flux;
    }
}
