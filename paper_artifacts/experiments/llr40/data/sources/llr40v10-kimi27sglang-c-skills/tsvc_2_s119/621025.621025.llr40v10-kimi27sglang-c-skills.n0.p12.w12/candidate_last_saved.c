#include <stdint.h>
#include <omp.h>

void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
  const int64_t B = 128;
  const int64_t NT = (LEN_2D + B - 1) / B;

  #pragma omp parallel
  for (int64_t td = 0; td <= 2 * (NT - 1); ++td) {
    #pragma omp for schedule(static)
    for (int64_t ti = (td < NT) ? 0 : td - (NT - 1);
         ti <= ((td < NT) ? td : NT - 1); ++ti) {
      const int64_t tj = td - ti;
      const int64_t i_start = ti * B;
      const int64_t i_end = (i_start + B < LEN_2D) ? i_start + B : LEN_2D;
      const int64_t j_start = tj * B;
      const int64_t j_end = (j_start + B < LEN_2D) ? j_start + B : LEN_2D;
      const int64_t i0 = (i_start < 1) ? 1 : i_start;
      const int64_t j0 = (j_start < 1) ? 1 : j_start;

      for (int64_t i = i0; i < i_end; ++i) {
        #pragma omp simd
        for (int64_t j = j0; j < j_end; ++j) {
          aa[i * LEN_2D + j] = aa[(i - 1) * LEN_2D + (j - 1)] + bb[i * LEN_2D + j];
        }
      }
    }
  }
}
