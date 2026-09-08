#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <omp.h>
#include <immintrin.h>

/* TSVC s323: b[i] = b[i-1] + t[i], t[i]=c[i]*(d[i]+e[i]); a[i] = b[i-1]+c[i]*d[i].
 * Scheme E:
 *  pass1: parallel streaming span totals (no scan, no latency chain)
 *  serial: prefix of the nt span totals
 *  pass2: single per-span ILP-2 vectorized scan with the span offset, final stores.
 * 64 B/elem DRAM traffic, 2 barriers.
 */
void tsvc_2_s323_fp64(double *restrict a, double *restrict b, const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
  const int64_t m = LEN_1D - 1;
  if (m <= 0) return;
  if (m < 4096) {
    for (int64_t i = 1; i < LEN_1D; ++i) {
      a[i] = b[i - 1] + c[i] * d[i];
      b[i] = a[i] + c[i] * e[i];
    }
    return;
  }

  int nt = omp_get_max_threads();
  if (nt > m) nt = (int)m;
  if (nt < 1) nt = 1;

  const int64_t bs = m / nt, rem = m % nt;
  double *tot = malloc(sizeof(double) * nt);
  if (!tot) {
    for (int64_t i = 1; i < LEN_1D; ++i) {
      a[i] = b[i - 1] + c[i] * d[i];
      b[i] = a[i] + c[i] * e[i];
    }
    return;
  }

  const __m512i IDX1 = _mm512_setr_epi64(0, 0, 1, 2, 3, 4, 5, 6);
  const __m512i IDX2 = _mm512_setr_epi64(0, 0, 0, 1, 2, 3, 4, 5);
  const __m512i IDX4 = _mm512_setr_epi64(0, 0, 0, 0, 0, 1, 2, 3);

#define SCAN8(t, s3)                                                               \
  do {                                                                             \
    const __m512d _sh1 = _mm512_maskz_permutex2var_pd(0xFE, (t), IDX1, (t));       \
    const __m512d _s1 = _mm512_add_pd((t), _sh1);                                  \
    const __m512d _sh2 = _mm512_maskz_permutex2var_pd(0xFC, _s1, IDX2, _s1);       \
    const __m512d _s2 = _mm512_add_pd(_s1, _sh2);                                  \
    const __m512d _sh4 = _mm512_maskz_permutex2var_pd(0xF0, _s2, IDX4, _s2);       \
    (s3) = _mm512_add_pd(_s2, _sh4);                                               \
  } while (0)

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int64_t s = (int64_t)tid * bs + (tid < rem ? tid : rem);
    const int64_t cnt = bs + (tid < rem);
    const double *cp = c + s + 1, *dp = d + s + 1, *ep = e + s + 1;
    double *ap = a + s + 1, *bp = b + s + 1;

    /* pass 1: span total, pure streaming (ILP-2) */
    double sum = 0.0;
    int64_t i = 0;
    for (; i + 16 <= cnt; i += 16) {
      const __m512d cA = _mm512_loadu_pd(cp + i), dA = _mm512_loadu_pd(dp + i), eA = _mm512_loadu_pd(ep + i);
      const __m512d cB = _mm512_loadu_pd(cp + i + 8), dB = _mm512_loadu_pd(dp + i + 8), eB = _mm512_loadu_pd(ep + i + 8);
      const __m512d tA = _mm512_mul_pd(cA, _mm512_add_pd(dA, eA));
      const __m512d tB = _mm512_mul_pd(cB, _mm512_add_pd(dB, eB));
      sum += _mm512_reduce_add_pd(tA) + _mm512_reduce_add_pd(tB);
    }
    for (; i + 8 <= cnt; i += 8) {
      const __m512d vc = _mm512_loadu_pd(cp + i), vd = _mm512_loadu_pd(dp + i), ve = _mm512_loadu_pd(ep + i);
      const __m512d t = _mm512_mul_pd(vc, _mm512_add_pd(vd, ve));
      sum += _mm512_reduce_add_pd(t);
    }
    for (; i < cnt; i++) sum += cp[i] * (dp[i] + ep[i]);
    tot[tid] = sum;

#pragma omp barrier
    if (tid == 0) {
      double run = b[0]; /* span t's carry-in = b[0] + all earlier spans */
      for (int t = 0; t < nt; t++) {
        const double x = tot[t];
        tot[t] = run;
        run += x;
      }
    }
#pragma omp barrier

    /* pass 2: span scan with offset, final stores (ILP-2)
     * NT stores where the 64B alignment allows: stream stores to a and/or b. */
    double carry = tot[tid]; /* includes b[0] for every span */
    const uintptr_t pha = (uintptr_t)ap & 63, phb = (uintptr_t)bp & 63;
    const int can_nt = ((pha - phb) & 63) == 0;
    i = 0;
    if (can_nt) {
      const int need = (int)(((64 - pha) & 63) >> 3); /* in doubles */
      for (i = 0; i < need && i < cnt; i++) {
        const double bprev = carry;
        const double ai = bprev + cp[i] * dp[i];
        const double bi = ai + cp[i] * ep[i];
        ap[i] = ai;
        bp[i] = bi;
        carry = bi;
      }
      for (; i + 16 <= cnt; i += 16) {
        const __m512d cA = _mm512_loadu_pd(cp + i), dA = _mm512_loadu_pd(dp + i), eA = _mm512_loadu_pd(ep + i);
        const __m512d cB = _mm512_loadu_pd(cp + i + 8), dB = _mm512_loadu_pd(dp + i + 8), eB = _mm512_loadu_pd(ep + i + 8);
        const __m512d tA = _mm512_mul_pd(cA, _mm512_add_pd(dA, eA));
        const __m512d cdA = _mm512_mul_pd(cA, dA);
        const __m512d tB = _mm512_mul_pd(cB, _mm512_add_pd(dB, eB));
        const __m512d cdB = _mm512_mul_pd(cB, dB);
        __m512d s3A, s3B;
        SCAN8(tA, s3A);
        SCAN8(tB, s3B);
        const double sumA = _mm512_reduce_add_pd(tA);
        const __m512d cvA = _mm512_set1_pd(carry);
        const __m512d cvB = _mm512_set1_pd(carry + sumA);
        const __m512d bA = _mm512_add_pd(cvA, s3A);
        _mm512_stream_pd(bp + i, bA);
        _mm512_stream_pd(ap + i, _mm512_add_pd(_mm512_sub_pd(bA, tA), cdA));
        const __m512d bB = _mm512_add_pd(cvB, s3B);
        _mm512_stream_pd(bp + i + 8, bB);
        _mm512_stream_pd(ap + i + 8, _mm512_add_pd(_mm512_sub_pd(bB, tB), cdB));
        carry += sumA + _mm512_reduce_add_pd(tB);
      }
      for (; i < cnt; i++) {
        const double bprev = carry;
        const double ai = bprev + cp[i] * dp[i];
        const double bi = ai + cp[i] * ep[i];
        ap[i] = ai;
        bp[i] = bi;
        carry = bi;
      }
    } else {
      for (i = 0; i + 16 <= cnt; i += 16) {
        const __m512d cA = _mm512_loadu_pd(cp + i), dA = _mm512_loadu_pd(dp + i), eA = _mm512_loadu_pd(ep + i);
        const __m512d cB = _mm512_loadu_pd(cp + i + 8), dB = _mm512_loadu_pd(dp + i + 8), eB = _mm512_loadu_pd(ep + i + 8);
        const __m512d tA = _mm512_mul_pd(cA, _mm512_add_pd(dA, eA));
        const __m512d cdA = _mm512_mul_pd(cA, dA);
        const __m512d tB = _mm512_mul_pd(cB, _mm512_add_pd(dB, eB));
        const __m512d cdB = _mm512_mul_pd(cB, dB);
        __m512d s3A, s3B;
        SCAN8(tA, s3A);
        SCAN8(tB, s3B);
        const double sumA = _mm512_reduce_add_pd(tA);
        const __m512d cvA = _mm512_set1_pd(carry);
        const __m512d cvB = _mm512_set1_pd(carry + sumA);
        const __m512d bA = _mm512_add_pd(cvA, s3A);
        _mm512_storeu_pd(bp + i, bA);
        _mm512_storeu_pd(ap + i, _mm512_add_pd(_mm512_sub_pd(bA, tA), cdA));
        const __m512d bB = _mm512_add_pd(cvB, s3B);
        _mm512_storeu_pd(bp + i + 8, bB);
        _mm512_storeu_pd(ap + i + 8, _mm512_add_pd(_mm512_sub_pd(bB, tB), cdB));
        carry += sumA + _mm512_reduce_add_pd(tB);
      }
      for (; i + 8 <= cnt; i += 8) {
        const __m512d vc = _mm512_loadu_pd(cp + i), vd = _mm512_loadu_pd(dp + i), ve = _mm512_loadu_pd(ep + i);
        const __m512d t = _mm512_mul_pd(vc, _mm512_add_pd(vd, ve));
        const __m512d cd = _mm512_mul_pd(vc, vd);
        __m512d s3;
        SCAN8(t, s3);
        const __m512d cv = _mm512_set1_pd(carry);
        const __m512d bv = _mm512_add_pd(cv, s3);
        _mm512_storeu_pd(bp + i, bv);
        _mm512_storeu_pd(ap + i, _mm512_add_pd(_mm512_sub_pd(bv, t), cd));
        carry += _mm512_reduce_add_pd(t);
      }
      for (; i < cnt; i++) {
        const double bprev = carry;
        const double ai = bprev + cp[i] * dp[i];
        const double bi = ai + cp[i] * ep[i];
        ap[i] = ai;
        bp[i] = bi;
        carry = bi;
      }
    }
  }
  free(tot);
}
