#define _GNU_SOURCE 1
#include <stdint.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>
extern int pthread_setaffinity_np(pthread_t, size_t, const cpu_set_t*);

static double now_ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec*1e9+ts.tv_nsec; }

static void mask_phys_only(cpu_set_t *set, int sock) {
  memset(set, 0, sizeof(*set));
  for (int c = sock*24; c < sock*24+24; ++c) set->__bits[c/64] |= 1UL << (c%64);
}

/* serial 4-socket placement probe of src (window: up to 1MB in the middle) */
static int detect_sock_serial(const double *src, int64_t n_elems) {
  int64_t w = n_elems < 131072 ? n_elems : 131072;
  int64_t n8 = w / 8;
  if (n8 < 8) return 0;
  int64_t base = (n_elems/2) - (n8*8)/2;
  double best = 1e30; int bs = 0;
  for (int sock = 0; sock < 4; ++sock) {
    cpu_set_t set; mask_phys_only(&set, sock);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    double t0 = now_ns();
    double acc = 0;
    for (int64_t i8 = 0; i8 < n8; ++i8)
      acc += _mm512_reduce_add_pd(_mm512_loadu_pd(src + base + i8*8));
    double t1 = now_ns();
    double score = (t1 - t0) + 1e9 * acc;
    if (score < best) { best = score; bs = sock; }
  }
  return bs;
}

static struct { const void *p; int sock; } reg[16];
static int nreg = 0;

static int sock_of(const double *p, int64_t n_elems) {
  for (int i = 0; i < nreg; ++i) if (reg[i].p == p) return reg[i].sock;
  int sock = detect_sock_serial(p, n_elems);
  if (nreg < 16) { reg[nreg].p = p; reg[nreg].sock = sock; ++nreg; }
  return sock;
}

/* work over [i0,i1): i0 must be a multiple of 8 when i1-i0 > 8 */
static void core(double *restrict a, const double *restrict b, double s, int64_t i0, int64_t i1) {
  const int64_t n8 = (i1 - i0) / 8;
  const __m256d vs = _mm256_set1_pd(s);
  #pragma omp for schedule(static)
  for (int64_t k = 0; k < n8; ++k) {
    double *ap = a + i0 + k*8;
    const double *bp = b + i0 + k*8;
    __m256d va1 = _mm256_loadu_pd(ap), vb1 = _mm256_loadu_pd(bp);
    __m256d va2 = _mm256_loadu_pd(ap+4), vb2 = _mm256_loadu_pd(bp+4);
    va1 = _mm256_fmadd_pd(vb1, vs, va1);
    va2 = _mm256_fmadd_pd(vb2, vs, va2);
    _mm256_storeu_pd(ap, va1);
    _mm256_storeu_pd(ap+4, va2);
  }
}

void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D, const int64_t S) {
  const double s = (double) S;
  const int64_t n = LEN_1D;
  if (n <= 0) return;
  if (n < (16LL << 20) / 8) { /* < 16MB per array: fits in cache, plain parallel */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) a[i] += b[i] * s;
    return;
  }
  const int sa = sock_of(a, n);
  const int sb = sock_of(b, n);
  if (sa == sb) {
    #pragma omp parallel num_threads(24)
    {
      cpu_set_t set; mask_phys_only(&set, sa);
      pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
      core(a, b, s, 0, (n/8)*8);
    }
  } else {
    const int64_t half = (n/2) / 8 * 8;
    #pragma omp parallel num_threads(24)
    {
      const int tid = omp_get_thread_num();
      cpu_set_t set; mask_phys_only(&set, tid < 12 ? sa : sb);
      pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
      if (tid < 12) core(a, b, s, 0, half);
      else          core(a, b, s, half, (n/8)*8);
    }
  }
  for (int64_t i = (n/8)*8; i < n; ++i) a[i] += b[i] * s;
}
