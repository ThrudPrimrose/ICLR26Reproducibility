#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_vag_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D){
  const int64_t n = LEN_1D;
  #pragma omp parallel
  {
    const int nt  = omp_get_num_threads();
    const int tid = omp_get_thread_num();
    const int64_t lo = n * tid / nt;
    const int64_t hi = n * (tid + 1) / nt;
    const int64_t B = 1024;
    int64_t base = lo;
    while (base < hi) {
      int64_t e  = base + B; if (e > hi) e = hi;
      int64_t e2 = e + B;    if (e2 > hi) e2 = hi;
      for (int64_t i = e; i < e2; ++i)
        _mm_prefetch((const char*)(b + ip[i]), _MM_HINT_T1);
      for (int64_t i = base; i < e; ++i)
        a[i] = b[ip[i]];
      base = e;
    }
  }
}
