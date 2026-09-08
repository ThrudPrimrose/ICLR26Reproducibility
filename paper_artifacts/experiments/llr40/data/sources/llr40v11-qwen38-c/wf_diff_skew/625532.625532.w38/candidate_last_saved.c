/* wf_diff_skew: a[i,j] += a[i-1,j] + a[i-1,j+1], rows sequential, j independent.
 *
 * Design: persistent pthread workers, one per core (affinity-limited).
 * Static column strips per worker; per-row fine-grained handshake: worker t's
 * last element of row i needs a[i-1][J1[t]] = worker t+1's FIRST element of row
 * i-1, so t publishes its first element of each row via a release-store on a
 * private flag and spins (near-zero iterations in steady state) on the right
 * neighbor's flag. No global per-row barrier. Rightmost strip reads a[i-1][m]
 * which is never written -> no sync.
 */
#include <stdint.h>
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>

#define MAXT 128

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_t g_tid[MAXT];
static int g_nt = 0;
static int g_cpu[MAXT];

static volatile unsigned long g_call;
static _Atomic unsigned long g_arrive;
static _Alignas(128) _Atomic unsigned long g_rowdone[MAXT];
static int64_t g_J0[MAXT + 1];
static int64_t g_n, g_m, g_B;
static const double *g_a;

static void *worker(void *arg)
{
  const int t = (int)(intptr_t)arg;
  if (g_cpu[t] >= 0) {
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(g_cpu[t], &cs);
    sched_setaffinity(0, sizeof(cs), &cs);
  }
  unsigned long last = 0;
  for (;;) {
    unsigned long target;
    do {
      target = __atomic_load_n(&g_call, __ATOMIC_ACQUIRE);
      __builtin_ia32_pause();
    } while (target == last);
    last = target;

    const int64_t n = g_n, m = g_m, js = g_J0[t], je = g_J0[t + 1];
    const double *restrict a = g_a;
    for (int64_t i = 1; i < n; ++i) {
      double *restrict cur = a + i * n;
      const double *restrict prv = cur - n;
      cur[js] = cur[js] + prv[js] + prv[js + 1];
      __atomic_store_n(&g_rowdone[t], (unsigned long)i, __ATOMIC_RELEASE);
      for (int64_t j = js + 1; j < je - 1; ++j)
        cur[j] = cur[j] + prv[j] + prv[j + 1];
      if (t != g_B - 1) {
        const unsigned long want = (unsigned long)(i - 1);
        unsigned long d = __atomic_load_n(&g_rowdone[t + 1], __ATOMIC_ACQUIRE);
        while (d < want) {
          __builtin_ia32_pause();
          d = __atomic_load_n(&g_rowdone[t + 1], __ATOMIC_ACQUIRE);
        }
      }
      cur[je - 1] = cur[je - 1] + prv[je - 1] + prv[je];
    }
    __atomic_add_fetch(&g_arrive, 1, __ATOMIC_RELEASE);
  }
}

static int affinity_count(void)
{
  cpu_set_t cs;
  CPU_ZERO(&cs);
  if (sched_getaffinity(0, sizeof(cs), &cs) != 0) return 1;
  return CPU_COUNT(&cs);
}

static void ensure_workers(int B)
{
  if (g_nt >= B) return;
  pthread_mutex_lock(&g_mu);
  if (g_nt >= B) { pthread_mutex_unlock(&g_mu); return; }
  cpu_set_t cs; CPU_ZERO(&cs);
  sched_getaffinity(0, sizeof(cs), &cs);
  int c = 0;
  for (int i = g_nt; i < B; ++i) {
    int cpu = -1;
    for (int k = 0; k < (int)CPU_SETSIZE; ++k)
      if (CPU_ISSET(k, &cs)) { cpu = k; break; }
    g_cpu[i] = cpu;
    if (pthread_create(&g_tid[i], NULL, worker, (void *)(intptr_t)i) != 0) { g_nt = i; break; }
    g_nt = i + 1;
    (void)c;
  }
  pthread_mutex_unlock(&g_mu);
}

static void run_serial(double *restrict a, const int64_t n, const int64_t m)
{
  for (int64_t i = 1; i < n; ++i) {
    double *restrict cur = a + i * n;
    const double *restrict prv = cur - n;
    for (int64_t j = 0; j < m; ++j)
      cur[j] = cur[j] + prv[j] + prv[j + 1];
  }
}

void wf_diff_skew_fp64(double *restrict a, const int64_t LEN_2D)
{
  const int64_t n = LEN_2D;
  if (n <= 1) return;
  const int64_t m = n - 1;

  int B = affinity_count();
  const char *e = getenv("WFS_THREADS");
  if (!e) e = getenv("OMP_NUM_THREADS");
  if (e) { int v = atoi(e); if (v > 0) B = v; }
  if (B > affinity_count()) B = affinity_count();
  if (B > MAXT) B = MAXT;
  if (B < 2 || m < 256) { run_serial(a, n, m); return; }
  if (B > (int)(m / 8)) B = (int)(m / 8);

  ensure_workers(B);

  for (int t = 0; t < B; ++t)
    __atomic_store_n(&g_rowdone[t], 0, __ATOMIC_RELAXED);
  __atomic_store_n(&g_arrive, 0, __ATOMIC_RELAXED);

  int64_t w = m / B;
  for (int t = 0; t <= B; ++t) g_J0[t] = (t == B) ? m : w * t;
  g_a = a; g_n = n; g_m = m; g_B = B;

  __atomic_store_n(&g_call, g_call + 1, __ATOMIC_RELEASE);
  while (__atomic_load_n(&g_arrive, __ATOMIC_ACQUIRE) < (unsigned long)B)
    __builtin_ia32_pause();
}
