/* TSVC tsvc_2 s1232, fp64.  Iteration space (spec): element (i,j) written iff j*VLEN<=i,
 * i.e. row i covers columns 0..i/VLEN.  aa write-only, bb/cc read-only (restrict) -> fully parallel.
 * AVX-512: aligned 64B loads (peeled) + non-temporal store for the write-once aa.
 * Block-cyclic row partition: thread t -> rows t, t+nt, t+2nt (all threads share a small
 * memory window per round -> better DRAM locality than per-thread contiguous spans). */
#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

__attribute__((target("avx512f")))
static void row_nt(double *a, const double *b, const double *c, int64_t n) {
  int64_t j = 0;
  while (j < n && (((uintptr_t)a & 63) != 0)) { *a++ = *b++ + *c++; j++; }
  if (j + 8 <= n && (((uintptr_t)b & 63) == 0) && (((uintptr_t)c & 63) == 0)) {
    for (; j + 8 <= n; j += 8) {
      _mm512_stream_pd(a, _mm512_add_pd(_mm512_load_pd(b), _mm512_load_pd(c)));
      a += 8; b += 8; c += 8;
    }
  }
  for (; j < n; j++) *a++ = *b++ + *c++;
}

void tsvc_2_s1232_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D,
                       const int64_t VLEN) {
  const int64_t L = LEN_2D, V = VLEN;
  if (L <= 0) return;
  if (V <= 0) {
    const int64_t n = L * L;
    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < n; ++k) aa[k] = bb[k] + cc[k];
    return;
  }
  int64_t nt = omp_get_max_threads();
  if (nt < 1) nt = 1;
  if (nt > L) nt = L;
  #pragma omp parallel num_threads(nt)
  {
    const int64_t me = omp_get_thread_num();
    for (int64_t i = me; i < L; i += nt)
      row_nt(aa + i * L, bb + i * L, cc + i * L, i / V + 1);
  }
}
