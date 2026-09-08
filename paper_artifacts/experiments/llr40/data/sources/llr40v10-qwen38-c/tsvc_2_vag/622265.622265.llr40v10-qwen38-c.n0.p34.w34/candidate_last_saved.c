#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

/* Gather with 4x in-flight 512-bit gathers and non-temporal stores:
   each b line is fetched once per element (random read floor), a is written
   streaming (no line fetch / no cache pollution). */
#define G8NT(IP,AP) do{ __m256i _i=_mm256_loadu_si256((const __m256i*)(IP)); \
  __m512i _o=_mm512_slli_epi64(_mm512_cvtepi32_epi64(_i),3); _mm512_stream_pd(AP,_mm512_i64gather_pd(_o,b,1)); }while(0)
#define G8N(I) do{ __m256i _i=_mm256_loadu_si256((const __m256i*)(ip+(I))); \
  __m512i _o=_mm512_slli_epi64(_mm512_cvtepi32_epi64(_i),3); _mm512_storeu_pd(a+(I),_mm512_i64gather_pd(_o,b,1)); }while(0)

void tsvc_2_vag_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip,
                     const int64_t n, uint8_t *ws, const int64_t ws_bytes)
{ (void)ws; (void)ws_bytes;
  const int nt = omp_get_max_threads();
  if (nt <= 1 || n < 262144) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) G8N(i);
    for (; i < n; ++i) a[i] = b[(size_t)ip[i]];
    return;
  }
  #pragma omp parallel num_threads(nt)
  {
    const int64_t chunk = (n + nt - 1) / nt;
    int64_t s0 = (int64_t)omp_get_thread_num() * chunk;
    int64_t e = s0 + chunk; if (e > n) e = n;
    int64_t s = s0;
    while (s < e && (s & 7)) { a[s] = b[(size_t)ip[s]]; ++s; }
    int64_t i = s;
    const int32_t *ipr = ip;
    double *ar = a;
    for (; i + 32 <= e; i += 32) { G8NT(ipr+i,ar+i); G8NT(ipr+i+8,ar+i+8); G8NT(ipr+i+16,ar+i+16); G8NT(ipr+i+24,ar+i+24); }
    for (; i + 8 <= e; i += 8) G8NT(ipr+i,ar+i);
    for (; i < e; ++i) a[i] = b[(size_t)ip[i]];
    _mm_sfence(); /* retire WC (non-temporal) stores before other cores read a */
  }
}
