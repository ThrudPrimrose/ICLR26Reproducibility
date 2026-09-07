// hpcagent_bench-autogen -- generated from heat_3d_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

/* 3-D 7-point heat diffusion, ping-pong between A and B.
 *
 * Each step:
 *   B[i] = alpha*(A[i+x]+A[i-x]+A[i+y]+A[i-y]+A[i+z]+A[i-z] - 6*A[i]) + A[i]
 *   A[i] = alpha*(B[i+x]+B[i-x]+B[i+y]+B[i-y]+B[i+z]+B[i-z] - 6*B[i]) + B[i]
 * for interior points only; boundaries are never touched (same as the NumPy
 * reference).  The arithmetic keeps the reference's order of operations:
 *   term = (P - 2.0*c) + Q   for each axis pair
 *   out  = ((alpha*tx + alpha*ty) + alpha*tz) + c
 *
 * Implementation: 2-D blocked (i,j tiles, full k line streamed) so the x/y
 * halo of a tile stays in cache and each thread drives a single DRAM stream.
 * The k loop is the AVX-512 vectorized inner loop.
 */
static void stencil_pass(const double *restrict S, double *restrict D,
                         int64_t N, int64_t N2, double alpha) {
    const int64_t M = N - 2;              /* interior extent in i and j      */
    const int64_t TI = 12, TJ = 12;       /* tile in i, j (fits ~1 MB L2)    */
    const int64_t nti = (M + TI - 1) / TI;
    const int64_t ntj = (M + TJ - 1) / TJ;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t ti = 0; ti < nti; ++ti) {
      for (int64_t tj = 0; tj < ntj; ++tj) {
        const int64_t i0 = 1 + ti * TI, i1 = i0 + TI < N - 1 ? i0 + TI : N - 1;
        const int64_t j0 = 1 + tj * TJ, j1 = j0 + TJ < N - 1 ? j0 + TJ : N - 1;
        for (int64_t i = i0; i < i1; ++i) {
          for (int64_t j = j0; j < j1; ++j) {
            for (int64_t k = 1; k < N - 1; ++k) {
              const int64_t idx = i * N2 + j * N + k;
              const double c  = S[idx];
              const double tx = (S[idx + N2] - 2.0 * c) + S[idx - N2];
              const double ty = (S[idx + N]  - 2.0 * c) + S[idx - N];
              const double tz = (S[idx + 1]  - 2.0 * c) + S[idx - 1];
              D[idx] = ((alpha * tx + alpha * ty) + alpha * tz) + c;
            }
          }
        }
      }
    }
}

void heat_3d_fp64(double *restrict A, double *restrict B, const int64_t N, const int64_t TSTEPS, const double alpha) {
    const int64_t N2 = N * N;
    for (int64_t t = 0; t < TSTEPS; ++t) {
        stencil_pass(A, B, N, N2, alpha);
        stencil_pass(B, A, N, N2, alpha);
    }
}
