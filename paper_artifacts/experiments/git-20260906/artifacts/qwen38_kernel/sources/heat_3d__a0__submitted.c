/* Heat-3D stencil (PolyBench heat_3d), interior-only 7-point update,
 * ping-pong between A and B. Exact expression of the numpy oracle:
 *   B = ((alpha*((ip-2c)+im) + alpha*((jp-2c)+jm)) + alpha*((kp-2c)+km)) + c
 */
#include <stdint.h>

static inline void stencil_pass(const double *restrict S, double *restrict D,
                                int64_t N, double alpha) {
    #pragma omp for collapse(2) schedule(static)
    for (int64_t i = 1; i < N - 1; ++i) {
        for (int64_t j = 1; j < N - 1; ++j) {
            const double *c  = S + (i * N + j) * N + 1;
            const double *ip = S + ((i + 1) * N + j) * N + 1;
            const double *im = S + ((i - 1) * N + j) * N + 1;
            const double *jp = S + (i * N + (j + 1)) * N + 1;
            const double *jm = S + (i * N + (j - 1)) * N + 1;
            double *b = D + (i * N + j) * N + 1;
            for (int64_t k = 1; k < N - 1; ++k) {
                const double cc = c[k - 1];
                const double t1 = alpha * ((ip[k - 1] - 2.0 * cc) + im[k - 1]);
                const double t2 = alpha * ((jp[k - 1] - 2.0 * cc) + jm[k - 1]);
                const double t3 = alpha * ((c[k] - 2.0 * cc) + c[k - 2]);
                b[k - 1] = ((t1 + t2) + t3) + cc;
            }
        }
    }
}

void heat_3d_fp64(double *restrict A, double *restrict B, int64_t N,
                  int64_t TSTEPS, double alpha) {
    if (N < 3 || TSTEPS < 1) return;
    #pragma omp parallel
    for (int64_t t = 0; t < TSTEPS; ++t) {
        stencil_pass(A, B, N, alpha);
        stencil_pass(B, A, N, alpha);
    }
}
