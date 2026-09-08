#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

void tsvc_2_s115_fp64(double *restrict a, const double *restrict aa, const int64_t N) {
  const int64_t B = 512;
  const int64_t nblocks = (N + B - 1) / B;
  #pragma omp parallel
  {
    const int T = omp_get_num_threads();
    const int tid = omp_get_thread_num();
    for (int64_t p = 0; p < nblocks; p++) {
      const int64_t js = p * B;
      int64_t je = (p + 1) * B; if (je > N) je = N;
      /* Step 2: within-block forward solve, serial over j (leader only) */
      if (tid == 0) {
        for (int64_t j = js; j < je; j++) {
          const double aj = a[j];
          const double *restrict row = aa + j * N;
          for (int64_t i = j + 1; i < je; i++)
            a[i] -= row[i] * aj;
        }
      }
      #pragma omp barrier
      /* Step 3: update future columns [je,N) using final rows [js,je) */
      const int64_t fs = je;
      const int64_t len = N - fs;
      if (len > 0) {
        const int64_t chunk = (len + T - 1) / T;
        const int64_t rs = fs + (int64_t)tid * chunk;
        int64_t re = rs + chunk; if (re > N) re = N;
        if (re > rs) {
          for (int64_t j = js; j < je; j++) {
            const double aj = a[j];
            const __m512d ajv = _mm512_set1_pd(aj);
            const double *restrict row = aa + j * N;
            if (j + 1 < je) {
              const double *nrow = aa + (j + 1) * N;
              __builtin_prefetch(nrow + rs, 0, 0);
              __builtin_prefetch(nrow + rs + 256, 0, 0);
              __builtin_prefetch(nrow + rs + 512, 0, 0);
            }
            int64_t i = rs;
            const int64_t iend = re & ~7LL;
            for (; i + 31 < re; i += 32) {
              __m512d x0 = _mm512_loadu_pd(row + i);
              __m512d x1 = _mm512_loadu_pd(row + i + 8);
              __m512d x2 = _mm512_loadu_pd(row + i + 16);
              __m512d x3 = _mm512_loadu_pd(row + i + 24);
              _mm512_storeu_pd(a + i,     _mm512_fnmadd_pd(x0, ajv, _mm512_loadu_pd(a + i)));
              _mm512_storeu_pd(a + i + 8, _mm512_fnmadd_pd(x1, ajv, _mm512_loadu_pd(a + i + 8)));
              _mm512_storeu_pd(a + i + 16,_mm512_fnmadd_pd(x2, ajv, _mm512_loadu_pd(a + i + 16)));
              _mm512_storeu_pd(a + i + 24,_mm512_fnmadd_pd(x3, ajv, _mm512_loadu_pd(a + i + 24)));
            }
            for (; i < iend; i += 8) {
              __m512d x = _mm512_loadu_pd(row + i);
              _mm512_storeu_pd(a + i, _mm512_fnmadd_pd(x, ajv, _mm512_loadu_pd(a + i)));
            }
            for (; i < re; i++) a[i] -= row[i] * aj;
          }
        }
      }
      #pragma omp barrier
    }
  }
}
