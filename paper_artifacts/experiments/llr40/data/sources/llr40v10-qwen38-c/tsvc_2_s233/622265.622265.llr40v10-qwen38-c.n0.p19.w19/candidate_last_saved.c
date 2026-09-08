#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

static __attribute__((always_inline)) inline void zmm_ntstore_pd(void *p, __m512d v) {
  __asm__ volatile ("vmovntpd %0, %1" :: "v"(v), "m"(*(double(*)[8])p) : "memory");
}

static const long idx1[8] = {0,0,1,2,3,4,5,6};
static const long idx2[8] = {0,0,0,1,2,3,4,5};
static const long idx4[8] = {0,0,0,0,0,1,2,3};

void tsvc_2_s233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
  const int64_t L = LEN_2D;
  const int64_t n = L - 8;
  if (n <= 0) return;
  const __m512i i1 = _mm512_loadu_si512(idx1);
  const __m512i i2 = _mm512_loadu_si512(idx2);
  const __m512i i4 = _mm512_loadu_si512(idx4);

  #pragma omp parallel
  {
    const int nt = omp_get_num_threads();
    const int tid = omp_get_thread_num();

    // ---- Part A: independent per column; 8-column zmm running sum, 6 streams per thread
    {
      const int64_t nfull = n / 8;
      const int64_t rem = n - 8 * nfull;
      const int SA = 6;
      const int64_t nslots = (int64_t)nt * SA;
      for (int64_t blk = (int64_t)tid * SA; blk < nfull; blk += nslots) {
        int64_t b[SA];
        int act = 0;
        for (int s = 0; s < SA; ++s)
          if (blk + s < nfull) { b[act] = blk + s; ++act; }
        double *ap[SA];
        const double *cp[SA];
        __m512d run[SA];
        for (int s = 0; s < act; ++s) {
          ap[s] = aa + 7 * L + 8 + 8 * b[s];
          cp[s] = cc + 8 + 8 * b[s];
          run[s] = _mm512_loadu_pd(ap[s]);
        }
        for (int64_t j = 0; j < L - 8; ++j) {
          const int64_t off0 = (j + 8) * L;
          const int64_t off1 = (j + 10) * L;
          const int64_t woff = (j + 1) * L;
          #pragma GCC unroll 6
          for (int s = 0; s < act; ++s) {
            run[s] = _mm512_add_pd(run[s], _mm512_loadu_pd(cp[s] + off0));
            __builtin_prefetch(cp[s] + off1, 0, 3);
            _mm512_storeu_pd(ap[s] + woff, run[s]);
          }
        }
      }
      if (rem > 0 && tid == 0) {
        const int64_t i0 = 8 * nfull;
        double *a0 = aa + 7 * L + 8 + i0;
        double run[8];
        for (int64_t t = 0; t < rem; ++t) run[t] = a0[t];
        for (int64_t j = 8; j < L; ++j) {
          const double *c = cc + j * L + 8 + i0;
          double *o = a0 + (j - 7) * L;
          for (int64_t t = 0; t < rem; ++t) { run[t] += c[t]; o[t] = run[t]; }
        }
      }
    }

    // ---- Part B: independent per row; zmm prefix with carry + 64B-aligned NT stores
    {
      const int CH = 4;
      for (int64_t j0 = 8 + tid; j0 < L; j0 += (int64_t)nt * CH) {
        const int64_t valid = (L - 1 - j0) / nt + 1;
        const int act = (valid < CH) ? (int)valid : CH;
        double carry[CH];
        const double *c[CH];
        double *o[CH];
        int64_t kc[CH];
        for (int h = 0; h < act; ++h) {
          int64_t j = j0 + (int64_t)h * nt;
          carry[h] = bb[j * L + 7];
          c[h] = cc + j * L + 8;
          o[h] = bb + j * L + 8;
          kc[h] = (8 - ((j * L + 8) % 8)) % 8;   // element offset of 64B-aligned run
          if (kc[h] > n) kc[h] = n;
          for (int64_t i = 0; i < kc[h]; ++i) { carry[h] += c[h][i]; o[h][i] = carry[h]; }
        }
        const int64_t nblk = (n + 7) / 8;
        for (int64_t k = 0; k < nblk; ++k) {
          #pragma GCC unroll 4
          for (int h = 0; h < act; ++h) {
            if (kc[h] + 8 > n) continue;
            __m512d p = _mm512_load_pd(c[h] + kc[h]);
            __m512d a = _mm512_add_pd(p, _mm512_maskz_permutex2var_pd(0xFE, p, i1, p));
            __m512d b = _mm512_add_pd(a, _mm512_maskz_permutex2var_pd(0xFC, a, i2, a));
            p = _mm512_add_pd(b, _mm512_maskz_permutex2var_pd(0xF0, b, i4, b));
            p = _mm512_add_pd(p, _mm512_set1_pd(carry[h]));
            zmm_ntstore_pd(o[h] + kc[h], p);
            double buf[8];
            _mm512_storeu_pd(buf, p);
            carry[h] = buf[7];
            kc[h] += 8;
          }
        }
        for (int h = 0; h < act; ++h) {
          for (; kc[h] < n; ++kc[h]) { carry[h] += c[h][kc[h]]; o[h][kc[h]] = carry[h]; }
        }
      }
    }
  }
}
