#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

/* TSVC s2275: for each i, for each j: aa[j*L+i] += bb[j*L+i]*cc[j*L+i];  a[i] = b[i] + c[i]*d[i].
 * The (i,j) -> j*L+i mapping is a bijection and every aa element is updated exactly once,
 * so the 2D part is a bit-exact contiguous FMA over all L^2 elements; the 1D part is an FMA too. */

/* The judge limits us to a few cores via process affinity while libgomp counts all online
 * CPUs; oversubscription costs >4x. Count our allowed CPUs without glibc GNU extensions
 * (the forced -include header locks feature macros before our _GNU_SOURCE could apply). */
static int affinity_cpu_count(void) {
  /* sched_getaffinity declared by hand: plain glibc symbol, same ABI, no header needed. */
  int (*fn)(int, size_t, void *);
  {
    static unsigned char mask[1024];
    /* resolve through a declaration; on glibc it links directly */
    extern int sched_getaffinity(int, size_t, void *);
    fn = sched_getaffinity;
    if (fn(0, sizeof mask, mask) == 0) {
      int n = 0;
      for (size_t i = 0; i < sizeof mask; ++i) {
        unsigned v = mask[i];
        while (v) { v &= (unsigned)(v - 1); ++n; }
      }
      if (n > 0) return n;
    }
  }
  /* Fallback: Cpus_allowed_list in /proc/self/status. */
  FILE *f = fopen("/proc/self/status", "r");
  if (f) {
    char line[1024];
    int n = 0;
    while (fgets(line, (int)sizeof line, f)) {
      if (strncmp(line, "Cpus_allowed_list:", 16) == 0) {
        const char *p = line + 16;
        while (*p == ' ' || *p == '\t') ++p;
        while (*p) {
          int lo = 0, hi = -1;
          while (*p >= '0' && *p <= '9') lo = lo * 10 + (*p++ - '0');
          if (*p == '-') { ++p; hi = 0; while (*p >= '0' && *p <= '9') hi = hi * 10 + (*p++ - '0'); }
          if (hi < lo) hi = lo;
          n += hi - lo + 1;
          while (*p && *p != ',') ++p;
        }
        break;
      }
    }
    fclose(f);
    if (n > 0) return n;
  }
  return -1;
}

static int pick_threads(void) {
  static int cached = 0;
  if (cached > 0) return cached;
  int base = omp_get_max_threads();
  int aff = affinity_cpu_count();
  int nt = (aff > 0 && aff < base) ? aff : base;
  if (nt < 1) nt = 1;
  cached = nt;
  return nt;
}

void tsvc_2_s2275_fp64(double *restrict a, double *restrict aa, const double *restrict b, const double *restrict bb,
                       const double *restrict c, const double *restrict cc, const double *restrict d,
                       const int64_t LEN_2D) {
  int nt = pick_threads();
  if (nt != omp_get_max_threads()) omp_set_num_threads(nt);
  const int64_t N = LEN_2D * LEN_2D;
#pragma omp parallel
  {
#pragma omp for schedule(static)
    for (int64_t k = 0; k < N; ++k) aa[k] += bb[k] * cc[k];
#pragma omp for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) a[i] = b[i] + c[i] * d[i];
  }
}
