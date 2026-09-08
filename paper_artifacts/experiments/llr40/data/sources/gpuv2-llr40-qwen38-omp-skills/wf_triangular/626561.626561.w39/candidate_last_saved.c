#include <stdint.h>
#include <omp.h>

void wf_triangular_fp64(double *restrict a, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N < 2) return;
  int64_t G = (N/2)/64*64; if (G < 64) G = 64; if (G > 77824) G = 77824;
  #pragma omp target parallel num_threads((int)G) map(tofrom: a[0:N*N])
  {
    const int tid  = omp_get_thread_num();
    const int Gact = omp_get_num_threads();
    for (int64_t d = 2; d <= 2*N - 2; d++) {
      int64_t iLo = d - N + 1; if (iLo < 1) iLo = 1;
      int64_t iHi = d / 2;
      const int64_t count = iHi - iLo + 1;
      for (int64_t e = tid; e < count; e += Gact) {
        const int64_t i = iLo + e;
        const int64_t j = d - i;
        a[i*N + j] = a[i*N + j] + a[(i-1)*N + j] + a[i*N + j - 1];
      }
      #pragma omp barrier
    }
  }
}
