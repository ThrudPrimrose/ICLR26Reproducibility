#include <stdint.h>

void tsvc_2_s275_fp64(double *restrict aa, const double *restrict bb,
                      const double *restrict cc, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  const int64_t M = N - 1;
  const int64_t K = 16;
  const int64_t L = (M + K - 1) / K;

  /* Pass 1: serial fold of each chunk; chunk 0 seeded with a0,
     others zero-based. Partial sums overwrite aa (j>=1 only).
     Slot J_c = min(M,(c+1)*L) then holds: c=0 -> P_1 = a0+S_0,
     c>=1 -> S_c (zero-based chunk sum). */
  #pragma omp target map(to: aa[0:N * N], bb[0:N * N], cc[0:N * N])
  {
    #pragma omp teams distribute parallel for
    for (int64_t t = 0; t < K * N; t++) {
      const int64_t c = t / N;
      const int64_t i = t - c * N;
      if (aa[i] <= 0.0) continue;
      const int64_t j0 = c * L + 1;
      int64_t j1 = (c + 1) * L;
      if (j1 > M) j1 = M;
      if (j0 > j1) continue;
      double acc = (c == 0) ? aa[i] : 0.0;
      for (int64_t j = j0; j <= j1; j++) {
        int64_t idx = j * N + i;
        acc += bb[idx] * cc[idx];
        aa[idx] = acc;
      }
    }
  }
  /* Carry prefix, one thread per column: slot J_c <- P_{c+1}. */
  #pragma omp target
  {
    #pragma omp teams distribute parallel for
    for (int64_t i = 0; i < N; i++) {
      if (aa[i] <= 0.0) continue;
      for (int64_t c = 1; c < K; c++) {
        if (c * L + 1 > M) break;
        int64_t jc = (c + 1) * L;
        if (jc > M) jc = M;
        const double S = aa[jc * N + i];
        aa[jc * N + i] = aa[c * L * N + i] + S;
      }
    }
  }
  /* Pass 2: add carry P_c to the non-final slots of chunks c>=1. */
  #pragma omp target
  {
    #pragma omp teams distribute parallel for
    for (int64_t t = N; t < K * N; t++) {
      const int64_t c = t / N;
      const int64_t i = t - c * N;
      if (aa[i] <= 0.0) continue;
      int64_t jc = (c + 1) * L;
      if (jc > M) jc = M;
      if (c * L + 1 >= jc) continue;
      const double carry = aa[c * L * N + i];
      for (int64_t j = c * L + 1; j < jc; j++) {
        int64_t idx = j * N + i;
        aa[idx] += carry;
      }
    }
  }
  #pragma omp target map(from: aa[0:N * N])
  { volatile double z = 0.0; (void)z; }
}
