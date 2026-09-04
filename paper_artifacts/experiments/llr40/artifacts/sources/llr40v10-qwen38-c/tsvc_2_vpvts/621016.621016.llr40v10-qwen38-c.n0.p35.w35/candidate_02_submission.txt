/* TSVC tsvc_2 kernel vpvts:  a[i] += b[i] * S   (fp64)
 *
 *  - small LEN: single-thread auto-vectorized loop (no parallel-region overhead)
 *  - large LEN + 64B-aligned a: AVX-512 with non-temporal stores (streams a,
 *    avoids read-for-ownership of the working set), manual equal split over
 *    threads, one _mm_sfence per thread
 *  - otherwise: OpenMP auto-vectorized loop
 */
#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

#define NT_THRESHOLD 33554432LL /* 32M elements = 256MB total footprint */
#define OMP_THRESHOLD 32768LL

void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D, const int64_t S) {
  if (LEN_1D <= 0) return;
  const double s = (double)S;

  if (LEN_1D <= OMP_THRESHOLD) {
    for (int64_t i = 0; i < LEN_1D; ++i) a[i] += b[i] * s;
    return;
  }

  if (LEN_1D >= NT_THRESHOLD && (((uintptr_t)a & 63) == 0)) {
    const __m512d vs = _mm512_set1_pd(s);
    const int64_t nvec = LEN_1D & ~15LL;
    const int64_t nb = nvec >> 7;      /* full 128-element blocks  */
    const int64_t nblk = nb << 7;      /* elements covered by vector part */
    #pragma omp parallel
    {
      const int nt = omp_get_num_threads();
      const int t  = omp_get_thread_num();
      int64_t lo = ((int64_t)t * nb) / nt;
      int64_t hi = ((int64_t)(t + 1) * nb) / nt;
      for (int64_t blk = lo; blk < hi; ++blk) {
        double *pa = a + (blk << 7);
        const double *pb = b + (blk << 7);
        for (int j = 0; j < 16; j += 2) {
          __m512d v0b = _mm512_loadu_pd(pb + 8 * j);
          __m512d v1b = _mm512_loadu_pd(pb + 8 * j + 8);
          __m512d v0a = _mm512_loadu_pd(pa + 8 * j);
          __m512d v1a = _mm512_loadu_pd(pa + 8 * j + 8);
          _mm512_stream_pd(pa + 8 * j,     _mm512_fmadd_pd(v0b, vs, v0a));
          _mm512_stream_pd(pa + 8 * j + 8, _mm512_fmadd_pd(v1b, vs, v1a));
        }
      }
      _mm_sfence();
    }
    for (int64_t i = nblk; i < LEN_1D; ++i) a[i] += b[i] * s;
    return;
  }

  #pragma omp parallel for schedule(static, 8192)
  for (int64_t i = 0; i < LEN_1D; ++i) a[i] += b[i] * s;
}
