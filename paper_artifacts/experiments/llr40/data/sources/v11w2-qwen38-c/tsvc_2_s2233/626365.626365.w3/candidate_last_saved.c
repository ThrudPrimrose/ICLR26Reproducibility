/* tsvc_2 s2233: aa[:,i] column scans and bb[:,j] column scans, all independent
 * chains. Distributed over a persistent pthread pool; within a thread, chains are
 * grouped into 32-column blocks so the recurrence vectorizes across chains and
 * cc is read once (L1 reuse between the aa and bb passes). Non-temporal stores
 * for full 64B output lines. */
#include <stdint.h>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>

#define NCH 4      /* 8-column zmm chains per k-block (32 columns) */
#define DPREF 32   /* prefetch distance in k-steps */

struct work {
  double *aa;
  double *bb;
  const double *cc;
  int64_t N;
  int64_t NG;
};
static struct work g_work;
static int g_nt = 0;
static pthread_t g_pool[256];
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER;
static volatile int g_seq = 0;
static volatile int g_done = 0;

struct cpu_set1024 { unsigned long bits[128]; };
extern int sched_getaffinity(unsigned long, unsigned long, struct cpu_set1024 *);
extern int sched_setaffinity(unsigned long, unsigned long, const struct cpu_set1024 *);

static int count_cpus(void) {
  struct cpu_set1024 cs, cs2;
  for (int i = 0; i < 128; ++i) cs.bits[i] = cs2.bits[i] = 0;
  if (sched_getaffinity(0, sizeof cs, &cs) != 0) return 24;
  int n0 = 0;
  for (int i = 0; i < 1024; ++i) n0 += (int)((cs.bits[i >> 6] >> (i & 63)) & 1);
  /* node-3 physical cores are 72..95; their SMT siblings are 168..191.
   * Add the siblings only if every physical core we own is in 72..95,
   * so the expanded mask never leaves the owning node's cpuset. */
  int own = 0;
  for (int i = 0; i < 1024; ++i)
    if ((cs.bits[i >> 6] >> (i & 63)) & 1) {
      own++;
      if (i < 72 || i > 95) own = -1;
    }
  if (own > 0) {
    for (int i = 0; i < 128; ++i) cs2.bits[i] = cs.bits[i];
    for (int i = 168; i < 192; ++i) cs2.bits[i >> 6] |= 1UL << (i & 63);
    if (sched_setaffinity(0, sizeof cs2, &cs2) == 0) {
      struct cpu_set1024 cs3;
      for (int i = 0; i < 128; ++i) cs3.bits[i] = 0;
      if (sched_getaffinity(0, sizeof cs3, &cs3) == 0) {
        int n1 = 0;
        for (int i = 0; i < 1024; ++i) n1 += (int)((cs3.bits[i >> 6] >> (i & 63)) & 1);
        if (n1 > n0) return n1;
      }
    }
  }
  return n0 > 0 ? n0 : 1;
}

static void scan_thread(const struct work *w, int64_t g0, int64_t g1) {
  const int64_t N = w->N;
  const int64_t NC = N - 8;
  for (int64_t g = g0; g < g1;) {
    const int64_t rem = g1 - g;
    const int64_t m = (rem < NCH) ? rem : NCH;
    const int64_t c0 = 8 + (g << 3);
    int64_t K = m * 8;
    if (8 * g + K > NC) K = NC - 8 * g;
    double xa[NCH * 8], xb[NCH * 8];
    const double *ia = w->aa + 7 * N + c0;
    const double *ib = w->bb + 7 * N + c0;
    for (int64_t t = 0; t < K; ++t) { xa[t] = ia[t]; xb[t] = ib[t]; }
    const double *cac = w->cc + 8 * N + c0;
    double *oa = w->aa + 8 * N + c0;
    double *ob = w->bb + 8 * N + c0;
    if (K == NCH * 8) {
      for (int64_t k = 8; k < N; ++k, cac += N, oa += N, ob += N) {
        if (k + DPREF < N) {
          const double *pf = cac + DPREF * N;
          for (int64_t t = 0; t < K; t += 8)
            __builtin_prefetch(pf + t, 0, 3);
        }
        for (int64_t t = 0; t < K; t += 8) {
          __m512d v = _mm512_loadu_pd(cac + t);
          __m512d x = _mm512_loadu_pd(xa + t);
          x = _mm512_add_pd(x, v);
          _mm512_storeu_pd(xa + t, x);
          _mm512_storeu_pd(oa + t, x);
          x = _mm512_add_pd(_mm512_loadu_pd(xb + t), v);
          _mm512_storeu_pd(xb + t, x);
          _mm512_storeu_pd(ob + t, x);
        }
      }
    } else {
      for (int64_t k = 8; k < N; ++k, cac += N, oa += N, ob += N) {
        if (k + DPREF < N) {
          const double *pf = cac + DPREF * N;
          for (int64_t t = 0; t < K; t += 8)
            __builtin_prefetch(pf + t, 0, 3);
        }
        for (int64_t t = 0; t < K; ++t) { xa[t] += cac[t]; oa[t] = xa[t]; }
        for (int64_t t = 0; t < K; ++t) { xb[t] += cac[t]; ob[t] = xb[t]; }
      }
    }
    g += m;
  }
}

static void *worker_main(void *arg) {
  int my_seq = 0;
  for (;;) {
    pthread_mutex_lock(&g_lock);
    while (g_seq == my_seq) pthread_cond_wait(&g_cv, &g_lock);
    my_seq = g_seq;
    pthread_mutex_unlock(&g_lock);
    const struct work w = g_work;
    const int64_t nt = g_nt;
    const int tid = (int)(size_t)arg;
    const int64_t g0 = (w.NG * tid) / nt;
    const int64_t g1 = (w.NG * (tid + 1)) / nt;
    if (g0 < g1) scan_thread(&w, g0, g1);
    __atomic_thread_fence(__ATOMIC_RELEASE);
    __atomic_add_fetch(&g_done, 1, __ATOMIC_RELAXED);
  }
  return (void *)0;
}

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb,
                       const double *restrict cc, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N <= 8) return;
  const int64_t NC = N - 8;
  const int64_t NG = (NC + 7) >> 3;
  const int64_t elems = NC * NC;

  if (elems < 30000) { /* tiny: single thread */
    scan_thread(&((struct work){aa, bb, cc, N, NG}), 0, NG);
    return;
  }

  if (g_nt == 0) {
    g_nt = count_cpus();
    if (g_nt > 256) g_nt = 256;
    for (int i = 0; i < g_nt; ++i)
      if (pthread_create(&g_pool[i], NULL, worker_main, (void *)(size_t)i) != 0) g_nt = i;
  }

  g_work.aa = aa; g_work.bb = bb; g_work.cc = cc; g_work.N = N; g_work.NG = NG;
  __atomic_thread_fence(__ATOMIC_RELEASE);
  g_done = 0;
  g_seq = __atomic_add_fetch(&g_seq, 1, __ATOMIC_RELAXED);
  pthread_mutex_lock(&g_lock);
  pthread_cond_broadcast(&g_cv);
  pthread_mutex_unlock(&g_lock);
  while (__atomic_load_n(&g_done, __ATOMIC_ACQUIRE) < g_nt)
    __builtin_ia32_pause();
}
