/* tsvc_2_s235 optimized (AVX2, custom persistent thread pool).
 *
 * Math, identical op order per element to the reference (bit-exact):
 *   a[i]    = a[i] + b[i]*c[i]         (single fma)
 *   aa[j,i] = aa[j-1,i] + bb[j,i]*a[i] (serial in j, independent across i)
 *
 * Columns are grouped in 32-column tiles. The serial j-chain per column keeps
 * its accumulator in a register, so row 0 of aa is the only aa read.  Each
 * worker streams 256 B of bb per row and writes 256 B of aa; when N is a
 * multiple of 4 the aa rows are 32 B-aligned and non-temporal stores avoid
 * read-allocate traffic.  A persistent worker pool (created at load time)
 * keeps every call warm.
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <immintrin.h>
#include <pthread.h>
#include <unistd.h>
#include <sched.h>
#include <sys/syscall.h>

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa, const double *restrict b,
                      const double *restrict bb, const double *restrict c, const int64_t LEN_2D);

#define TILE 32
#define PF 96

#define FUTEX_WAIT 0
#define FUTEX_WAKE 1
static int futex_op(volatile int *addr, int op, int val) {
  return (int)syscall(SYS_futex, addr, op, val, 0, 0, 0);
}

/* ---------------- tiles ---------------- */
static inline void scan_tile_full(double *restrict a, double *restrict aa,
                                  const double *restrict b, const double *restrict bb,
                                  const double *restrict c, int64_t col, int64_t N, int nt) {
  const double *bb0 = bb + col;
  double *aa0 = aa + col;
  __m256d av[8], acc[8];
  for (int g = 0; g < 8; ++g) {
    const int64_t o = g * 4;
    av[g] = _mm256_loadu_pd(a + col + o);
    av[g] = _mm256_fmadd_pd(_mm256_loadu_pd(b + col + o), _mm256_loadu_pd(c + col + o), av[g]);
    _mm256_storeu_pd(a + col + o, av[g]);
    acc[g] = _mm256_loadu_pd(aa0 + o);
  }
  for (int64_t j = 1; j < N; ++j) {
    const double *bp = bb0 + j * N;
    if (j + PF < N) _mm_prefetch((const char *)(bb0 + (j + PF) * N), _MM_HINT_T1);
    __m256d v0 = _mm256_loadu_pd(bp);
    __m256d v1 = _mm256_loadu_pd(bp + 4);
    __m256d v2 = _mm256_loadu_pd(bp + 8);
    __m256d v3 = _mm256_loadu_pd(bp + 12);
    __m256d v4 = _mm256_loadu_pd(bp + 16);
    __m256d v5 = _mm256_loadu_pd(bp + 20);
    __m256d v6 = _mm256_loadu_pd(bp + 24);
    __m256d v7 = _mm256_loadu_pd(bp + 28);
    acc[0] = _mm256_fmadd_pd(v0, av[0], acc[0]);
    acc[1] = _mm256_fmadd_pd(v1, av[1], acc[1]);
    acc[2] = _mm256_fmadd_pd(v2, av[2], acc[2]);
    acc[3] = _mm256_fmadd_pd(v3, av[3], acc[3]);
    acc[4] = _mm256_fmadd_pd(v4, av[4], acc[4]);
    acc[5] = _mm256_fmadd_pd(v5, av[5], acc[5]);
    acc[6] = _mm256_fmadd_pd(v6, av[6], acc[6]);
    acc[7] = _mm256_fmadd_pd(v7, av[7], acc[7]);
    double *op = aa0 + j * N;
    if (nt) {
      _mm256_stream_pd(op, acc[0]);
      _mm256_stream_pd(op + 4, acc[1]);
      _mm256_stream_pd(op + 8, acc[2]);
      _mm256_stream_pd(op + 12, acc[3]);
      _mm256_stream_pd(op + 16, acc[4]);
      _mm256_stream_pd(op + 20, acc[5]);
      _mm256_stream_pd(op + 24, acc[6]);
      _mm256_stream_pd(op + 28, acc[7]);
    } else {
      _mm256_storeu_pd(op, acc[0]);
      _mm256_storeu_pd(op + 4, acc[1]);
      _mm256_storeu_pd(op + 8, acc[2]);
      _mm256_storeu_pd(op + 12, acc[3]);
      _mm256_storeu_pd(op + 16, acc[4]);
      _mm256_storeu_pd(op + 20, acc[5]);
      _mm256_storeu_pd(op + 24, acc[6]);
      _mm256_storeu_pd(op + 28, acc[7]);
    }
  }
  _mm_sfence();
}

static inline void scan_tile_tail(double *restrict a, double *restrict aa,
                                  const double *restrict b, const double *restrict bb,
                                  const double *restrict c, int64_t col, int64_t N, int64_t w) {
  __mmask8 mg[8];
  const __mmask8 mf = 0xFu;
  for (int g = 0; g < 8; ++g)
    mg[g] = (w > g * 4) ? mf : 0u;
  int r = (int)(w & 3);
  if (r)
    mg[w / 4 - 1] = (__mmask8)((1u << r) - 1u);
  const double *bb0 = bb + col;
  double *aa0 = aa + col;
  __m256d av[8], acc[8];
  for (int g = 0; g < 8; ++g) {
    const int64_t o = g * 4;
    if (mg[g]) {
      av[g] = _mm256_maskz_loadu_pd(mg[g], a + col + o);
      av[g] = _mm256_fmadd_pd(_mm256_maskz_loadu_pd(mg[g], b + col + o),
                              _mm256_maskz_loadu_pd(mg[g], c + col + o), av[g]);
      _mm256_mask_storeu_pd(a + col + o, mg[g], av[g]);
      acc[g] = _mm256_maskz_loadu_pd(mg[g], aa0 + o);
    } else {
      av[g] = _mm256_setzero_pd();
      acc[g] = _mm256_setzero_pd();
    }
  }
  for (int64_t j = 1; j < N; ++j) {
    const double *bp = bb0 + j * N;
    if (j + PF < N) _mm_prefetch((const char *)(bb0 + (j + PF) * N), _MM_HINT_T1);
    for (int g = 0; g < 8; ++g)
      if (mg[g])
        acc[g] = _mm256_fmadd_pd(_mm256_maskz_loadu_pd(mg[g], bp + g * 4), av[g], acc[g]);
    double *op = aa0 + j * N;
    for (int g = 0; g < 8; ++g)
      if (mg[g])
        _mm256_mask_storeu_pd(op + g * 4, mg[g], acc[g]);
  }
  _mm_sfence();
}

static inline void do_tile(double *restrict a, double *restrict aa, const double *restrict b,
                           const double *restrict bb, const double *restrict c, int64_t col,
                           int64_t N, int nt) {
  int64_t w = N - col;
  if (w > TILE)
    w = TILE;
  if (w >= TILE)
    scan_tile_full(a, aa, b, bb, c, col, N, nt);
  else
    scan_tile_tail(a, aa, b, bb, c, col, N, w);
}

/* ---------------- persistent pool ---------------- */
#define MAXT 64

static volatile int g_state;          /* 0 idle, 1 working */
static volatile int g_claim;
static volatile int g_arr;
static volatile int g_expected;
static int g_pool_size;

static double *g_a, *g_aa;
static const double *g_b, *g_bb, *g_c;
static int64_t g_N;
static int g_ntiles;
static int g_nt;

static pthread_t g_pool[MAXT];

#define CLAIM(v) __atomic_fetch_add(&g_claim, 1, __ATOMIC_ACQ_REL)
#define ARRIVE() __atomic_fetch_add(&g_arr, 1, __ATOMIC_ACQ_REL)

static void worker_round(void) {
  for (;;) {
    int t = CLAIM(1);
    if (t >= g_ntiles)
      break;
    do_tile(g_a, g_aa, g_b, g_bb, g_c, (int64_t)t * TILE, g_N, g_nt);
  }
  int n = ARRIVE() + 1;
  if (n >= g_expected) {
    __atomic_store_n(&g_arr, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&g_state, 0, __ATOMIC_RELEASE);
  } else {
    while (__atomic_load_n(&g_state, __ATOMIC_ACQUIRE) != 0)
      __builtin_ia32_pause();
  }
}

static void *worker_main(void *arg) {
  (void)arg;
  for (;;) {
    for (;;) {
      if (__atomic_load_n(&g_state, __ATOMIC_ACQUIRE) == 1) {
        worker_round();
        break;
      }
      __builtin_ia32_pause();
    }
    for (int i = 0; i < 200000; i++) {
      if (__atomic_load_n(&g_state, __ATOMIC_ACQUIRE) == 1)
        break;
      __builtin_ia32_pause();
    }
    if (__atomic_load_n(&g_state, __ATOMIC_ACQUIRE) == 0)
      futex_op((volatile int *)&g_state, FUTEX_WAIT, 0);
  }
  return 0;
}

static int count_cpus(void) {
  cpu_set_t cs;
  if (sched_getaffinity(0, sizeof(cs), &cs) != 0)
    return 12;
  int n = 0;
  for (int i = 0; i < CPU_SETSIZE; ++i)
    if (CPU_ISSET(i, &cs))
      n++;
  return n ? n : 1;
}

__attribute__((constructor)) static void pool_init(void) {
  int cpus = count_cpus();
  g_pool_size = cpus > MAXT ? MAXT : (cpus < 1 ? 1 : cpus);
  for (int i = 0; i < g_pool_size; ++i)
    if (pthread_create(&g_pool[i], 0, worker_main, (void *)(intptr_t)i) != 0) {
      g_pool_size = i;
      break;
    }
}

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa, const double *restrict b,
                      const double *restrict bb, const double *restrict c, const int64_t LEN_2D) {
  if (LEN_2D <= 1)
    return;
  const int64_t ntiles = (LEN_2D + TILE - 1) / TILE;
  g_nt = (int)((LEN_2D & 3) == 0);

  if (ntiles <= 8) {
    for (int64_t t = 0; t < ntiles; ++t)
      do_tile(a, aa, b, bb, c, t * TILE, LEN_2D, g_nt);
    _mm_mfence();
    return;
  }

  g_a = a;
  g_aa = aa;
  g_b = b;
  g_bb = bb;
  g_c = c;
  g_N = LEN_2D;
  g_ntiles = (int)ntiles;
  __atomic_store_n(&g_claim, 0, __ATOMIC_RELAXED);
  __atomic_store_n(&g_arr, 0, __ATOMIC_RELAXED);
  __atomic_store_n(&g_expected, g_pool_size + 1, __ATOMIC_RELAXED);
  __atomic_store_n(&g_state, 1, __ATOMIC_RELEASE);
  futex_op((volatile int *)&g_state, FUTEX_WAKE, g_pool_size);

  worker_round(); /* dispatcher participates */
  _mm_mfence();
}
