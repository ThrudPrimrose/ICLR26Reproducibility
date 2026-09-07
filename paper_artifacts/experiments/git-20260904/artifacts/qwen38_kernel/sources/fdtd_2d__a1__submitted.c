/* FDTD 2D update -- explicit AVX-512, two waves per timestep:
 *   Wave 1: update ex and ey in place (both read only the OLD hz -> independent)
 *   Wave 2: update hz in place using the NEW ex/ey
 * ~6 reads + 3 writes per element per timestep (vs 11+3 for the naive 4-pass form). */
#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

#define TGT __attribute__((target("avx512f,avx512dq,fma,bmi2")))

static TGT inline void ey_row(double *restrict yr, const double *restrict hr, const double *restrict hp,
                              int64_t NY, double neyc) {
  __m512d vc = _mm512_set1_pd(neyc);
  int64_t j = 0;
  for (; j + 16 <= NY; j += 16) {
    __m512d a = _mm512_loadu_pd(hr + j), b = _mm512_loadu_pd(hp + j);
    __m512d a1 = _mm512_loadu_pd(hr + j + 8), b1 = _mm512_loadu_pd(hp + j + 8);
    __m512d y = _mm512_loadu_pd(yr + j);
    __m512d y1 = _mm512_loadu_pd(yr + j + 8);
    y = _mm512_fmadd_pd(vc, _mm512_sub_pd(a, b), y);
    y1 = _mm512_fmadd_pd(vc, _mm512_sub_pd(a1, b1), y1);
    _mm512_storeu_pd(yr + j, y);
    _mm512_storeu_pd(yr + j + 8, y1);
  }
  for (; j < NY; ++j) yr[j] += neyc * (hr[j] - hp[j]);
}

static TGT inline void ex_row(double *restrict xr, const double *restrict hr,
                              int64_t NY, double nexc) {
  __m512d vc = _mm512_set1_pd(nexc);
  if (NY <= 1) return;
  int64_t j = 1;
  for (; j + 16 <= NY; j += 16) {
    __m512d a = _mm512_loadu_pd(hr + j), ap = _mm512_loadu_pd(hr + j - 1);
    __m512d a1 = _mm512_loadu_pd(hr + j + 8), ap1 = _mm512_loadu_pd(hr + j + 7);
    __m512d x = _mm512_loadu_pd(xr + j);
    __m512d x1 = _mm512_loadu_pd(xr + j + 8);
    x = _mm512_fmadd_pd(vc, _mm512_sub_pd(a, ap), x);
    x1 = _mm512_fmadd_pd(vc, _mm512_sub_pd(a1, ap1), x1);
    _mm512_storeu_pd(xr + j, x);
    _mm512_storeu_pd(xr + j + 8, x1);
  }
  for (; j < NY; ++j) xr[j] += nexc * (hr[j] - hr[j - 1]);
}

static TGT inline void hz_row(double *restrict hr, const double *restrict x0,
                              const double *restrict y0, const double *restrict y1,
                              int64_t NY, double nhzc) {
  __m512d vc = _mm512_set1_pd(nhzc);
  int64_t m = NY - 1;
  if (m <= 0) return;
  int64_t j = 0;
  for (; j + 16 <= m; j += 16) {
    __m512d h = _mm512_loadu_pd(hr + j);
    __m512d h1 = _mm512_loadu_pd(hr + j + 8);
    __m512d x = _mm512_loadu_pd(x0 + j), xs = _mm512_loadu_pd(x0 + j + 1);
    __m512d x1 = _mm512_loadu_pd(x0 + j + 8), xs1 = _mm512_loadu_pd(x0 + j + 9);
    __m512d ya = _mm512_loadu_pd(y0 + j), yb = _mm512_loadu_pd(y1 + j);
    __m512d ya1 = _mm512_loadu_pd(y0 + j + 8), yb1 = _mm512_loadu_pd(y1 + j + 8);
    __m512d d = _mm512_add_pd(_mm512_sub_pd(xs, x), _mm512_sub_pd(yb, ya));
    __m512d d1 = _mm512_add_pd(_mm512_sub_pd(xs1, x1), _mm512_sub_pd(yb1, ya1));
    h = _mm512_fmadd_pd(vc, d, h);
    h1 = _mm512_fmadd_pd(vc, d1, h1);
    _mm512_storeu_pd(hr + j, h);
    _mm512_storeu_pd(hr + j + 8, h1);
  }
  for (; j < m; ++j) hr[j] += nhzc * ((x0[j + 1] - x0[j]) + (y1[j] - y0[j]));
}

void fdtd_2d_fp64(double *restrict ex, double *restrict ey, const double *restrict fict, double *restrict hz,
                  int64_t NX, int64_t NY, int64_t TMAX, double ex_courant, double ey_courant, double hz_courant) {
  const double nexc = -ex_courant, neyc = -ey_courant, nhzc = -hz_courant;
  if (TMAX <= 0 || NX <= 0 || NY <= 0) return;

  #pragma omp parallel
  {
    for (int64_t t = 0; t < TMAX; ++t) {
      const double ft = fict[t];

      /* Wave 1: ex/ey in place (old hz only) */
      #pragma omp for schedule(static)
      for (int64_t i = 0; i < NX; ++i) {
        double *xr = ex + i * NY;
        double *yr = ey + i * NY;
        const double *hr = hz + i * NY;
        if (i == 0) {
          __m512d v = _mm512_set1_pd(ft);
          int64_t j = 0;
          for (; j + 16 <= NY; j += 16) {
            _mm512_storeu_pd(yr + j, v);
            _mm512_storeu_pd(yr + j + 8, v);
          }
          for (; j < NY; ++j) yr[j] = ft;
        } else {
          ey_row(yr, hr, hz + (i - 1) * NY, NY, neyc);
        }
        ex_row(xr, hr, NY, nexc);
      }
      #pragma omp barrier

      /* Wave 2: hz in place with new ex/ey */
      #pragma omp for schedule(static)
      for (int64_t i = 0; i < NX - 1; ++i) {
        hz_row(hz + i * NY, ex + i * NY, ey + i * NY, ey + (i + 1) * NY, NY, nhzc);
      }
    }
  }
}
