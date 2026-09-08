/* TSVC tsvc_2 s2710 fp64 -- AVX-512 masked + OpenMP + NUMA-aware pinning.
 *
 * Reference semantics (per element i):
 *   if (a[i] > b[i]) {
 *     a[i] += b[i]*d[i];
 *     if (LEN_1D > 10) c[i] += d[i]*d[i]; else c[i] = d[i]*e[i] + 1.0;
 *   } else {
 *     b[i] = a[i] + e[i]*e[i];
 *     if (x[0] > 0.0) c[i] = a[i] + d[i]*d[i]; else c[i] += e[i]*e[i];
 *   }
 * Independent per i -> trivially parallel; branch made data-parallel with masks.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>

extern int sched_setaffinity(int pid, unsigned long cpusetsize, const unsigned long *mask);

#define MAXC 1024
typedef struct {
  int valid;
  int ncpus;
  unsigned long mask[MAXC / 64];
} pininfo_t;

static pininfo_t g_pin;
static const void *g_pin_key = (void *)1;
static int g_pin_init = 0;

/* Detect the NUMA node holding `key` (via /proc/self/numa_maps) and fill g_pin
 * with that node's CPU mask. Returns 1 if we can pin to a single node. */
static int pin_for(const void *key) {
  if (g_pin_init && g_pin_key == key)
    return g_pin.valid;
  memset(&g_pin, 0, sizeof g_pin);
  g_pin_key = key;
  g_pin_init = 1;
  unsigned long ka = (unsigned long)key;
  int node = -1;
  FILE *f = fopen("/proc/self/numa_maps", "r");
  if (f) {
    char line[2048];
    while (fgets(line, sizeof line, f)) {
      unsigned long s = 0, e = 0;
      if (sscanf(line, "%lx-%lx", &s, &e) != 2)
        continue;
      if (ka < s || ka >= e)
        continue;
      int ni = 0;
      char *q = line;
      while (*q) {
        if (q[0] == 'N' && q[1] >= '0' && q[1] <= '9') {
          ni++;
          if (ni == 1)
            node = atoi(q + 1);
        }
        q++;
      }
      if (ni != 1)
        node = -1; /* pages interleave across nodes: no single node to pin to */
      break;
    }
    fclose(f);
  }
  if (node < 0)
    return 0;
  char path[64];
  snprintf(path, sizeof path, "/sys/devices/system/node/node%d/cpulist", node);
  f = fopen(path, "r");
  if (!f)
    return 0;
  char buf[4096];
  size_t got = fread(buf, 1, sizeof buf - 1, f);
  buf[got] = 0;
  fclose(f);
  char *tok = buf;
  while (*tok) {
    char *comma = strchr(tok, ',');
    if (comma)
      *comma = 0;
    unsigned long a = 0, b = 0;
    if (sscanf(tok, "%lu-%lu", &a, &b) == 2) {
      for (unsigned long i = a; i <= b && i < MAXC; ++i)
        g_pin.mask[i / 64] |= 1UL << (i & 63);
    } else if (sscanf(tok, "%lu", &a) == 1 && a < MAXC) {
      g_pin.mask[a / 64] |= 1UL << (a & 63);
    }
    tok = comma ? comma + 1 : buf + strlen(buf);
  }
  int ncp = 0;
  for (int w = 0; w < MAXC / 64; ++w)
    ncp += __builtin_popcountl(g_pin.mask[w]);
  g_pin.ncpus = ncp;
  g_pin.valid = ncp > 0;
  return g_pin.valid;
}

static inline void body8(double *a, double *b, double *c, const double *d, const double *e,
                         int small, int pos) {
  __m512d va = _mm512_loadu_pd(a);
  __m512d vb = _mm512_loadu_pd(b);
  __m512d vd = _mm512_loadu_pd(d);
  __m512d ve = _mm512_loadu_pd(e);
  __m512d vc = _mm512_loadu_pd(c);

  __mmask8 gt = _mm512_cmp_pd_mask(va, vb, _CMP_GT_OQ);
  __m512d bd = _mm512_mul_pd(vb, vd);
  __m512d ee = _mm512_mul_pd(ve, ve);
  __m512d dd = _mm512_mul_pd(vd, vd);

  __m512d na = _mm512_mask_add_pd(va, gt, va, bd);             /* a: += b*d where gt  */
  __m512d nb = _mm512_mask_add_pd(vb, (__mmask8)~gt, va, ee);  /* b: = a+e*e where !gt */

  __m512d nc_gt, nc_le;
  if (small)
    nc_gt = _mm512_add_pd(_mm512_mul_pd(vd, ve), _mm512_set1_pd(1.0));
  else
    nc_gt = _mm512_add_pd(vc, dd);
  if (pos)
    nc_le = _mm512_add_pd(va, dd);
  else
    nc_le = _mm512_add_pd(vc, ee);
  __m512d nc = _mm512_mask_mov_pd(nc_le, gt, nc_gt);

  _mm512_storeu_pd(a, na);
  _mm512_storeu_pd(b, nb);
  _mm512_storeu_pd(c, nc);
}

static inline void tail_scalar(double *a, double *b, double *c, const double *d, const double *e,
                               int64_t n0, int64_t n, int small, int pos) {
  for (int64_t i = n0; i < n; ++i) {
    if (a[i] > b[i]) {
      a[i] += b[i] * d[i];
      if (small)
        c[i] = d[i] * e[i] + 1.0;
      else
        c[i] += d[i] * d[i];
    } else {
      b[i] = a[i] + e[i] * e[i];
      if (pos)
        c[i] = a[i] + d[i] * d[i];
      else
        c[i] += e[i] * e[i];
    }
  }
}

void tsvc_2_s2710_fp64(double *restrict a, double *restrict b, double *restrict c,
                       const double *restrict d, const double *restrict e,
                       const double *restrict x, const int64_t LEN_1D) {
  const int small = (LEN_1D <= 10) ? 1 : 0;
  const int pos = (x[0] > 0.0) ? 1 : 0;
  const int64_t n = LEN_1D;
  const int64_t nv = (n / 8) * 8;

  if (n >= 262144 && pin_for(a)) {
    int maxt = omp_get_max_threads();
    int nt = maxt < g_pin.ncpus ? maxt : g_pin.ncpus;
    omp_set_num_threads(nt);
    const unsigned long *m = g_pin.mask;
    #pragma omp parallel num_threads(nt)
    {
      sched_setaffinity(0, (unsigned long)(MAXC / 64) * 8, m);
      #pragma omp for schedule(static)
      for (int64_t i0 = 0; i0 < nv; i0 += 8)
        body8(a + i0, b + i0, c + i0, d + i0, e + i0, small, pos);
    }
    tail_scalar(a, b, c, d, e, nv, n, small, pos);
    return;
  }

  if (nv >= 16384) {
    #pragma omp parallel for schedule(static)
    for (int64_t i0 = 0; i0 < nv; i0 += 8)
      body8(a + i0, b + i0, c + i0, d + i0, e + i0, small, pos);
  } else {
    for (int64_t i0 = 0; i0 < nv; i0 += 8)
      body8(a + i0, b + i0, c + i0, d + i0, e + i0, small, pos);
  }
  tail_scalar(a, b, c, d, e, nv, n, small, pos);
}
