// 2x2 (i,j) block, 4 outputs per k-loop iteration
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

static void
heat_3d_half(const double *restrict src, double *restrict dst,
             const int n, const double alpha)
{
    const int ni = n - 2;
    const long long nn = (long long)n * n;

    #pragma omp for schedule(static)
    for (int i = 1; i <= ni - 1; i += 2) {
        const double *r_im1 = src + ((long long)i - 1) * nn;
        const double *r_0   = src + (long long)i * nn;
        const double *r_1   = src + ((long long)i + 1) * nn;
        const double *r_2   = src + ((long long)i + 2) * nn;
        double *b_0 = dst + (long long)i * nn;
        double *b_1 = dst + ((long long)i + 1) * nn;

        for (int j = 1; j <= ni - 1; j += 2) {
            const long long jn = (long long)j * n;
            const double *l_im1 = r_im1 + jn;
            const double *l_0   = r_0   + jn;
            const double *l_1   = r_1   + jn;
            const double *l_2   = r_2   + jn;
            double *o0 = b_0 + jn;
            double *o1 = b_1 + jn;
            for (int k = 1; k <= ni; ++k) {
                const double ac00 = l_0[k];
                o0[k] = alpha * (l_1[k] - 2.0*ac00 + l_im1[k])
                      + alpha * (l_0[k+n] - 2.0*ac00 + l_0[k-n])
                      + alpha * (l_0[k+1] - 2.0*ac00 + l_0[k-1]) + ac00;

                const double ac01 = l_0[k+n];
                o0[k+n] = alpha * (l_1[k+n] - 2.0*ac01 + l_im1[k+n])
                        + alpha * (l_0[k+2*n] - 2.0*ac01 + ac00)
                        + alpha * (l_0[k+n+1] - 2.0*ac01 + l_0[k+n-1]) + ac01;

                const double ac10 = l_1[k];
                o1[k] = alpha * (l_2[k] - 2.0*ac10 + ac00)
                      + alpha * (l_1[k+n] - 2.0*ac10 + l_1[k-n])
                      + alpha * (l_1[k+1] - 2.0*ac10 + l_1[k-1]) + ac10;

                const double ac11 = l_1[k+n];
                o1[k+n] = alpha * (l_2[k+n] - 2.0*ac11 + ac01)
                        + alpha * (l_1[k+2*n] - 2.0*ac11 + ac10)
                        + alpha * (l_1[k+n+1] - 2.0*ac11 + l_1[k+n-1]) + ac11;
            }
        }
    }
    // odd tail row (i = ni) if ni odd
    if (ni & 1) {
        const int i = ni;
        const double *ri   = src + (long long)i * nn;
        const double *rip1 = src + ((long long)i + 1) * nn;
        const double *rim1 = src + ((long long)i - 1) * nn;
        double *bi         = dst + (long long)i * nn;
        for (int j = 1; j <= ni; ++j) {
            const double *rj   = ri   + (long long)j * n;
            const double *rjp  = rip1 + (long long)j * n;
            const double *rjm  = rim1 + (long long)j * n;
            double *o          = bi   + (long long)j * n;
            for (int k = 1; k <= ni; ++k) {
                const double ac = rj[k];
                o[k] = alpha * (rjp[k] - 2.0 * ac + rjm[k])
                     + alpha * (rj[k + n] - 2.0 * ac + rj[k - n])
                     + alpha * (rj[k + 1] - 2.0 * ac + rj[k - 1])
                     + ac;
            }
        }
    }
}

void heat_3d_fp64(double *restrict A, double *restrict B,
                  const int64_t N, const int64_t TSTEPS, const double alpha)
{
    if (N < 4 || TSTEPS < 1) return;
    const int n = (int)N;
    #pragma omp parallel
    for (int64_t t = 0; t < TSTEPS; ++t) {
        heat_3d_half(A, B, n, alpha);
        heat_3d_half(B, A, n, alpha);
    }
}
