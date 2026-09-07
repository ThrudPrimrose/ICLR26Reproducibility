#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#ifndef TI
#define TI 16
#endif
#ifndef TJ
#define TJ 16
#endif
#ifndef TK
#define TK 1024
#endif
static inline void do_tile(const double *restrict A, double *restrict B, int64_t NN, int n,
                           int i0, int j0, int k0, int i1, int j1, int k1, double alpha) {
  for (int i = i0; i < i1; i++) {
    const double *ci = A + (int64_t)i * NN;
    double *oi = B + (int64_t)i * NN;
    for (int j = j0; j < j1; j++) {
      const double *c = ci + j * n;
      double *o = oi + j * n;
      const double *px = c + NN, *mx = c - NN, *py = c + n, *my = c - n;
      for (int k = k0; k < k1; k++) {
        double cc = c[k];
        double t1 = alpha * ((px[k] - 2.0 * cc) + mx[k]);
        double t2 = alpha * ((py[k] - 2.0 * cc) + my[k]);
        double t3 = alpha * ((c[k + 1] - 2.0 * cc) + c[k - 1]);
        o[k] = ((t1 + t2) + t3) + cc;
      }
    }
  }
}
void heat_3d_fp64(double *restrict A, double *restrict B, const int64_t N, const int64_t TSTEPS, const double alpha) {
  const int64_t NN = N * N;
  const int n = (int)N;
  const int n1 = (int)(N - 1);
  for (int64_t t = 1; t <= TSTEPS; t++) {
    #pragma omp parallel for collapse(3) schedule(static)
    for (int64_t ti = 1; ti < n1; ti += TI)
    for (int64_t tj = 1; tj < n1; tj += TJ)
    for (int64_t tk = 1; tk < n1; tk += TK) {
      int i1 = (int)(ti + TI) < n1 ? (int)(ti + TI) : n1;
      int j1 = (int)(tj + TJ) < n1 ? (int)(tj + TJ) : n1;
      int k1 = (int)(tk + TK) < n1 ? (int)(tk + TK) : n1;
      do_tile(A, B, NN, n, (int)ti, (int)tj, (int)tk, i1, j1, k1, alpha);
    }
    #pragma omp parallel for collapse(3) schedule(static)
    for (int64_t ti = 1; ti < n1; ti += TI)
    for (int64_t tj = 1; tj < n1; tj += TJ)
    for (int64_t tk = 1; tk < n1; tk += TK) {
      int i1 = (int)(ti + TI) < n1 ? (int)(ti + TI) : n1;
      int j1 = (int)(tj + TJ) < n1 ? (int)(tj + TJ) : n1;
      int k1 = (int)(tk + TK) < n1 ? (int)(tk + TK) : n1;
      do_tile(B, A, NN, n, (int)ti, (int)tj, (int)tk, i1, j1, k1, alpha);
    }
  }
}
