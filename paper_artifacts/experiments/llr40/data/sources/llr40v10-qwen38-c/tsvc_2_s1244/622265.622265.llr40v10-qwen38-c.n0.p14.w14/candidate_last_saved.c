#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>

/* Count CPUs allowed to this process from /proc/self/status, clamped by cgroup CFS quota. */
static int detect_ncpu(void) {
  int n = 0;
  FILE *f = fopen("/proc/self/status", "r");
  if (f) {
    char line[4096];
    while (fgets(line, sizeof line, f)) {
      if (!strncmp(line, "Cpus_allowed_list:", 18)) {
        char *p = line + 18;
        while (*p) {
          while (*p && *p != ',' && *p != '-' && (*p < '0' || *p > '9')) p++;
          if (!*p) break;
          int lo = atoi(p);
          while (*p && (*p >= '0' && *p <= '9')) p++;
          if (*p == '-') {
            p++;
            int hi = atoi(p);
            n += hi - lo + 1;
          } else {
            n += 1;
          }
          while (*p && *p != ',') p++;
        }
        break;
      }
    }
    fclose(f);
  }
  if (n <= 0) n = 1;
  /* cgroup v2 cpu.max: "quota period" or "max period" */
  f = fopen("/sys/fs/cgroup/cpu.max", "r");
  if (f) {
    int quota, period;
    char c = 0;
    if (fscanf(f, "%d %d %c", &quota, &period, &c) == 2 && quota > 0 && period > 0) {
      int q = quota / period + (quota % period > 0 ? 1 : 0);
      if (q < n) n = q;
    }
    fclose(f);
  } else {
    /* cgroup v1 */
    f = fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r");
    FILE *fp = fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r");
    if (f && fp) {
      int quota, period;
      if (fscanf(f, "%d", &quota) == 1 && fscanf(fp, "%d", &period) == 1 && quota > 0 && period > 0) {
        int q = quota / period + (quota % period > 0 ? 1 : 0);
        if (q < n) n = q;
      }
    }
    if (f) fclose(f);
    if (fp) fclose(fp);
  }
  return n;
}

static int get_nt(void) {
  static int v = -1;
  if (v < 0) v = detect_ncpu();
  return v;
}

static void core_loop(double *restrict a, const double *restrict b, const double *restrict c,
                      double *restrict d, const int64_t S, const int64_t E, const int64_t phase,
                      const int skip_seam) {
  int64_t i = S;
  while (i < E) {
    const double f = b[i] + c[i] * c[i] + b[i] * b[i] + c[i];
    d[i] = f + a[i + 1];
    if (!skip_seam) a[i] = f;
    i++;
    if ((i & 7) == phase) break;
  }
  for (; i + 8 <= E; i += 8) {
    __m512d va = _mm512_loadu_pd(a + i + 1);
    __m512d vb = _mm512_loadu_pd(b + i);
    __m512d vc = _mm512_loadu_pd(c + i);
    __m512d f = _mm512_add_pd(vb, _mm512_mul_pd(vc, vc));
    f = _mm512_add_pd(f, _mm512_mul_pd(vb, vb));
    f = _mm512_add_pd(f, vc);
    _mm512_storeu_pd(a + i, f);
    _mm512_storeu_pd(d + i, _mm512_add_pd(f, va));
  }
  for (; i < E; i++) {
    const double f = b[i] + c[i] * c[i] + b[i] * b[i] + c[i];
    d[i] = f + a[i + 1];
    a[i] = f;
  }
}

static void seams(double *restrict a, const double *restrict b, const double *restrict c,
                  const int nt, const int64_t chunk) {
  (void)0;
  for (int T = 1; T < nt; T++) {
    const int64_t i = (int64_t)T * chunk;
    a[i] = b[i] + c[i] * c[i] + b[i] * b[i] + c[i];
  }
}

void tsvc_2_s1244_fp64(double *restrict a, const double *restrict b, const double *restrict c, double *restrict d,
                       const int64_t LEN_1D) {
  const int64_t nI = LEN_1D - 1; /* i in [0, nI-1] */
  if (nI <= 0) return;

  const int64_t aoff8 = (int64_t)(((uintptr_t)a & 63) >> 3);
  const int64_t phase = (8 - aoff8) & 7;

  const int g_nt = get_nt();
  if (g_nt <= 2) {
    core_loop(a, b, c, d, 0, nI, phase, 0);
    return;
  }
  int nt = g_nt;
  if ((int64_t)nt > nI) nt = (int)nI;
  const int64_t chunk = nI / nt;
#pragma omp parallel num_threads(nt)
  {
    const int tid = omp_get_thread_num();
    int64_t S = (int64_t)tid * chunk;
    int64_t E = (tid == nt - 1) ? nI : (int64_t)(tid + 1) * chunk;
    core_loop(a, b, c, d, S, E, phase, tid > 0);
  }
  seams(a, b, c, nt, chunk);
}
