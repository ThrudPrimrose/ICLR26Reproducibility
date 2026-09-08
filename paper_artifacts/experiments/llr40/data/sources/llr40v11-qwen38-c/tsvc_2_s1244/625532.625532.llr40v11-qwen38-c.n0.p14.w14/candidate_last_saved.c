#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

/* TSVC tsvc_2 s1244 -- see v2 notes: d[i] = a_new[i] + a_orig[i+1];
 * single pass, per-thread contiguous block, `saved` boundary value + barrier.
 * v3a: 8-wide (two AVX2 vectors) main loop for deeper load MLP. */
void tsvc_2_s1244_fp64(double *restrict a, const double *restrict b, const double *restrict c,
                       double *restrict d, const int64_t LEN_1D) {
  const int64_t n = LEN_1D - 1;
  if (n <= 0) return;

  #pragma omp parallel
  {
    const int nt  = omp_get_num_threads();
    const int tid = omp_get_thread_num();

    const int64_t block = n / nt;
    const int64_t rem   = n % nt;
    const int64_t i0 = tid * block + (tid < rem ? tid : rem);
    const int64_t len = block + (tid < rem ? 1 : 0);

    if (len > 0) {
      const int64_t i1 = i0 + len - 1;
      const double saved = a[i1 + 1];  /* original, read before any store */
      #pragma omp barrier
      int64_t i = i0;
      for (; i + 8 <= i1; i += 8) {
        const __m256d va0 = _mm256_loadu_pd(a + i);
        const __m256d va1 = _mm256_loadu_pd(a + i + 4);
        const __m256d vb0 = _mm256_loadu_pd(b + i);
        const __m256d vb1 = _mm256_loadu_pd(b + i + 4);
        const __m256d vc0 = _mm256_loadu_pd(c + i);
        const __m256d vc1 = _mm256_loadu_pd(c + i + 4);
        __m256d an0 = _mm256_mul_pd(vc0, vc0);
        an0 = _mm256_fmadd_pd(vb0, vb0, an0);
        an0 = _mm256_add_pd(an0, vb0);
        an0 = _mm256_add_pd(an0, vc0);
        __m256d an1 = _mm256_mul_pd(vc1, vc1);
        an1 = _mm256_fmadd_pd(vb1, vb1, an1);
        an1 = _mm256_add_pd(an1, vb1);
        an1 = _mm256_add_pd(an1, vc1);
        const double *const p0 = (const double *)&va0;
        const double *const p1 = (const double *)&va1;
        const __m256d next0 = _mm256_set_pd(p1[0], p0[3], p0[2], p0[1]);
        const __m256d next1 = _mm256_set_pd(a[i + 8], p1[3], p1[2], p1[1]);
        _mm256_storeu_pd(a + i, an0);
        _mm256_storeu_pd(a + i + 4, an1);
        _mm256_storeu_pd(d + i, _mm256_add_pd(an0, next0));
        _mm256_storeu_pd(d + i + 4, _mm256_add_pd(an1, next1));
      }
      for (; i + 4 <= i1; i += 4) {
        const __m256d va = _mm256_loadu_pd(a + i);
        const __m256d vb = _mm256_loadu_pd(b + i);
        const __m256d vc = _mm256_loadu_pd(c + i);
        __m256d an = _mm256_mul_pd(vc, vc);
        an = _mm256_fmadd_pd(vb, vb, an);
        an = _mm256_add_pd(an, vb);
        an = _mm256_add_pd(an, vc);
        const double *const p = (const double *)&va;
        const __m256d nxt = _mm256_set_pd(a[i + 4], p[3], p[2], p[1]);
        _mm256_storeu_pd(a + i, an);
        _mm256_storeu_pd(d + i, _mm256_add_pd(an, nxt));
      }
      for (; i <= i1; i++) {
        const double an = b[i] + c[i] * c[i] + b[i] * b[i] + c[i];
        a[i] = an;
        d[i] = an + (i + 1 <= i1 ? a[i + 1] : saved);
      }
    } else {
      #pragma omp barrier
    }
  }
}
