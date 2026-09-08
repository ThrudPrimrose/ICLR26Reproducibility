#include <stdint.h>
#include <omp.h>

#ifndef BH
#define BH 48
#endif
#ifndef BW
#define BW 48
#endif

void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N <= 1) return;

  const int64_t H = BH;
  const int64_t W = BW;
  const int64_t NTi = (N + H - 1) / H;
  const int64_t NTj = (N + W - 1) / W;
  const int64_t ndiag = NTi + NTj - 1;

  #pragma omp parallel
  {
    for (int64_t diag = 0; diag < ndiag; ++diag) {
      int64_t ti_start = (diag < NTi) ? 0 : diag - (NTi - 1);
      int64_t ti_end   = (diag < NTj) ? diag + 1 : NTi;
      if (ti_end > NTi) ti_end = NTi;

      #pragma omp for schedule(static)
      for (int64_t ti = ti_start; ti < ti_end; ++ti) {
        const int64_t tj = diag - ti;
        if (tj < 0 || tj >= NTj) continue;
        const int64_t i0 = ti * H;
        const int64_t i1 = (i0 + H < N) ? i0 + H : N;
        const int64_t j0 = tj * W;
        const int64_t j1 = (j0 + W < N) ? j0 + W : N;
        const int64_t i_start = (i0 < 1) ? 1 : i0;

        for (int64_t i = i_start; i < i1; ++i) {
          const double *restrict bb_row = bb + i * N;
          const double *restrict aa_prev = aa + (i - 1) * N;
          double *restrict aa_row = aa + i * N;
          const int64_t j_start = (j0 < 1) ? 1 : j0;

          #pragma omp simd
          for (int64_t j = j_start; j < j1; ++j) {
            aa_row[j] = aa_prev[j - 1] + bb_row[j];
          }
        }
      }
    }
  }
}
