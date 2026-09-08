/* TSVC tsvc_2_5 ext_break_capture: first i with a[i] > 1.0.
 * 64-bit loads only (wide SIMD memory ops are unreliable on this hardware).
 * Striped scalar scan across the CPUs the container actually allows. */
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

/* count CPUs in Cpus_allowed_list (cgroup may restrict to a NUMA node) */
static int64_t ec_cpu_count(void) {
  static int64_t cached = 0;
  if (cached > 0) return cached;
  int64_t cnt = 1;
  FILE *f = fopen("/proc/self/status", "r");
  if (f) {
    char line[4096];
    while (fgets(line, sizeof line, f)) {
      if (strncmp(line, "Cpus_allowed_list:", 16) == 0) {
        char *p = line + 16;
        while (*p == ' ' || *p == '\t') p++;
        while (*p) {
          char *end;
          long lo = strtol(p, &end, 10);
          long hi = lo;
          if (*end == '-') {
            hi = strtol(end + 1, &end, 10);
            if (hi < lo) hi = lo;
          }
          cnt += hi - lo + 1;
          if (*end == ',') end++;
          p = end;
        }
        break;
      }
    }
    fclose(f);
  }
  cached = cnt < 1 ? 1 : cnt;
  return cached;
}

void ext_break_capture_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D) {
  const double k = 1.0;
  const int64_t n = LEN_1D;
  if (n <= 0) { out_index[0] = -1; out_value[0] = -1.0; return; }

  int64_t best = INT64_MAX;
  const int64_t ncpu = ec_cpu_count();

  if (ncpu >= 2 && n >= 8192) {
    const int nt = (int)(ncpu > 32 ? 32 : ncpu);
    #pragma omp parallel num_threads(nt)
    {
      const int tid = omp_get_thread_num();
      const int64_t lo = (n * tid) / nt;
      const int64_t hi = (n * (tid + 1)) / nt;
      int64_t i = lo;
      for (; i + 8 <= hi; i += 8) {
        if (__atomic_load_n(&best, __ATOMIC_RELAXED) < i) break;
        const double d0 = a[i + 0], d1 = a[i + 1], d2 = a[i + 2], d3 = a[i + 3];
        const double d4 = a[i + 4], d5 = a[i + 5], d6 = a[i + 6], d7 = a[i + 7];
        const int mask = (int)(d0 > k) | ((int)(d1 > k) << 1) | ((int)(d2 > k) << 2) | ((int)(d3 > k) << 3) |
                          ((int)(d4 > k) << 4) | ((int)(d5 > k) << 5) | ((int)(d6 > k) << 6) | ((int)(d7 > k) << 7);
        if (mask) {
          const int64_t j = i + (int64_t)__builtin_ctz((unsigned)mask);
          int64_t cur = best;
          while (j < cur) {
            if (__atomic_compare_exchange_n(&best, &cur, j, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
              break;
          }
          break;
        }
      }
      for (; i < hi; ++i) {
        if (__atomic_load_n(&best, __ATOMIC_RELAXED) <= i) break;
        if (a[i] > k) {
          int64_t cur = best;
          while (i < cur) {
            if (__atomic_compare_exchange_n(&best, &cur, i, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
              break;
          }
          break;
        }
      }
    }
    if (best == INT64_MAX) best = -1;
  } else {
    best = -1;
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
      const double d0 = a[i + 0], d1 = a[i + 1], d2 = a[i + 2], d3 = a[i + 3];
      const double d4 = a[i + 4], d5 = a[i + 5], d6 = a[i + 6], d7 = a[i + 7];
      const int mask = (int)(d0 > k) | ((int)(d1 > k) << 1) | ((int)(d2 > k) << 2) | ((int)(d3 > k) << 3) |
                        ((int)(d4 > k) << 4) | ((int)(d5 > k) << 5) | ((int)(d6 > k) << 6) | ((int)(d7 > k) << 7);
      if (mask) { best = i + (int64_t)__builtin_ctz((unsigned)mask); break; }
    }
    for (; i < n; ++i)
      if (a[i] > k) { best = i; break; }
  }

  out_index[0] = best;
  out_value[0] = (best < 0) ? -1.0 : a[best];
}
