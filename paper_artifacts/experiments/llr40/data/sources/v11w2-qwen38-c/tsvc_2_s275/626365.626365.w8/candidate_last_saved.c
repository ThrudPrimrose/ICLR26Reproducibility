/* tsvc_2_s275: for each column i, if aa[0][i] > 0 then
   aa[j][i] = aa[j-1][i] + bb[j][i]*cc[j][i]  for j=1..N-1.
   Columns independent; j is a serial scan.  Parallelize over column blocks;
   loop j outermost and the thread's blocks inside (MLP: all blocks stream
   bb/cc at the same j).  AVX-512 masked ops keep the active() guard branch-free.
   NUMA: detect the data's node and pin threads to its local cores. */
#define _GNU_SOURCE
#include <stdint.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <unistd.h>
#include <sys/syscall.h>
#ifdef __AVX512F__
#include <immintrin.h>
#endif

#ifndef BLOCKCOLS
#define BLOCKCOLS 128
#endif

/* ---- NUMA detection + pinning (run once) ---- */
static int g_inited = 0;
static cpu_set_t g_pin;
static int g_pincnt = 0;

static int parse_cpulist(const char *s, cpu_set_t *out) {
  int n = 0;
  CPU_ZERO(out);
  const char *p = s;
  while (*p) {
    while (*p == ' ' || *p == ',' || *p == '\n') p++;
    if (!*p) break;
    char *end;
    long a = strtol(p, &end, 10);
    long b = a;
    if (*end == '-') { p = end + 1; b = strtol(p, &end, 10); }
    for (long c = a; c <= b && c < CPU_SETSIZE; c++) { CPU_SET((int)c, out); n++; }
    p = end;
  }
  return n;
}

static int detect_datanode(const double *aa) {
  (void)aa;
  FILE *f = fopen("/proc/self/numa_maps", "r");
  if (!f) return -1;
  long long sum[64] = {0};
  int maxnode = 0;
  char line[8192];
  while (fgets(line, sizeof(line), f)) {
    if (!strstr(line, "anon=")) continue;
    char *t = line;
    while (*t) {
      if (t[0] == 'N' && t[1] >= '0' && t[1] <= '9') {
        int node = t[1] - '0';
        if (node < 64) {
          t += 2;
          long long v = 0;
          while (*t >= '0' && *t <= '9') { v = v * 10 + (*t - '0'); t++; }
          sum[node] += v;
          if (node > maxnode) maxnode = node;
        }
      }
      t++;
    }
  }
  fclose(f);
  long long best = -1; int bestnode = -1;
  for (int n = 0; n <= maxnode && n < 64; n++)
    if (sum[n] > best) { best = sum[n]; bestnode = n; }
  return bestnode;
}

static void setup_pinning(const double *aa) {
  if (g_inited) return;
  g_inited = 1;
  CPU_ZERO(&g_pin);
  if (sched_getaffinity(0, sizeof(g_pin), &g_pin) != 0) return;
  g_pincnt = CPU_COUNT(&g_pin);
  int node = detect_datanode(aa);
  if (node < 0) return; /* keep all allowed CPUs */
  char path[64];
  snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
  FILE *f = fopen(path, "r");
  if (!f) return;
  char buf[1024] = "";
  size_t r = fread(buf, 1, sizeof(buf) - 1, f);
  (void)r;
  fclose(f);
  cpu_set_t nodecpus;
  parse_cpulist(buf, &nodecpus);
  cpu_set_t both;
  CPU_ZERO(&both);
  for (int c = 0; c < CPU_SETSIZE; c++)
    if (CPU_ISSET(c, &nodecpus) && CPU_ISSET(c, &g_pin)) CPU_SET(c, &both);
  int cnt = CPU_COUNT(&both);
  if (cnt > 0) { g_pin = both; g_pincnt = cnt; }
}

void tsvc_2_s275_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  const int64_t nblocks = (N + BLOCKCOLS - 1) / BLOCKCOLS;

  setup_pinning(aa);
  if (g_pincnt > 0) {
    cpu_set_t p = g_pin;
    sched_setaffinity(0, sizeof(p), &p);
    omp_set_num_threads(g_pincnt);
  }

#ifdef __AVX512F__
  #pragma omp parallel
  {
    const int nt = omp_get_num_threads();
    const int tid = omp_get_thread_num();
    const int64_t per = nblocks / nt;
    const int extra = (int)(nblocks - per * nt);
    const int64_t b0 = per * tid + (tid < extra ? tid : extra);
    const int64_t nb = per + (tid < extra ? 1 : 0);
    const int nbl = (nb > 0) ? (int)nb : 1;
    __mmask8 mv[nbl][16], ma[nbl][16];
    int nsg_arr[nbl];
    int64_t c0_arr[nbl];
    for (int64_t li = 0; li < nb; li++) {
      const int64_t b = b0 + li, c0 = b * BLOCKCOLS;
      const int64_t c1 = c0 + BLOCKCOLS < N ? c0 + BLOCKCOLS : N;
      c0_arr[li] = c0;
      const int nsg = (int)((c1 - c0 + 7) / 8);
      nsg_arr[li] = nsg;
      for (int sg = 0; sg < nsg; sg++) {
        const int64_t c = c0 + (int64_t)sg * 8;
        __mmask8 m = 0, a = 0;
        for (int l = 0; l < 8; l++) {
          const int64_t i = c + l;
          if (i < N) { m |= (__mmask8)(1u << l); if (aa[i] > 0.0) a |= (__mmask8)(1u << l); }
        }
        mv[li][sg] = m; ma[li][sg] = a;
      }
    }
    for (int64_t j = 1; j < N; j++) {
      for (int64_t li = 0; li < nb; li++) {
        const int64_t c0 = c0_arr[li];
        const int nsg = nsg_arr[li];
        for (int sg = 0; sg < nsg; sg++) {
          if (ma[li][sg] == 0) continue;
          const int64_t c = c0 + (int64_t)sg * 8;
          double *ap = aa + j * N + c;
          const double *apm1 = aa + (j - 1) * N + c;
          const double *bp = bb + j * N + c;
          const double *cp = cc + j * N + c;
          __m512d v, bv, cv;
          if (mv[li][sg] == 0xFF) { v = _mm512_loadu_pd(apm1); bv = _mm512_loadu_pd(bp); cv = _mm512_loadu_pd(cp); }
          else { v = _mm512_mask_loadu_pd(_mm512_setzero_pd(), mv[li][sg], apm1);
                 bv = _mm512_mask_loadu_pd(_mm512_setzero_pd(), mv[li][sg], bp);
                 cv = _mm512_mask_loadu_pd(_mm512_setzero_pd(), mv[li][sg], cp); }
          v = _mm512_fmadd_pd(bv, cv, v);
          if (ma[li][sg] == 0xFF) _mm512_storeu_pd(ap, v);
          else _mm512_mask_storeu_pd(ap, ma[li][sg], v);
        }
      }
    }
  }
#else
  #pragma omp parallel for schedule(static)
  for (int64_t b = 0; b < nblocks; b++) {
    const int64_t c0 = b * BLOCKCOLS;
    const int64_t c1 = c0 + BLOCKCOLS < N ? c0 + BLOCKCOLS : N;
    for (int64_t j = 1; j < N; j++)
      for (int64_t i = c0; i < c1; i++)
        if (aa[i] > 0.0) aa[j * N + i] = aa[(j - 1) * N + i] + bb[j * N + i] * cc[j * N + i];
  }
#endif
}
