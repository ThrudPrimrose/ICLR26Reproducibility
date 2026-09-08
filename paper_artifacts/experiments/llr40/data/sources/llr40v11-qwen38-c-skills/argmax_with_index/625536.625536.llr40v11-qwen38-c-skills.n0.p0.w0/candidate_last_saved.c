/* argmax_with_index (TSVC tsvc_2_5) - SIMD multi-core version.
 *
 * Reference semantics: left-to-right scan with strict '>', so the FIRST
 * occurrence of the maximum wins. Merging two left-to-right ordered ranges:
 * the right range wins only when strictly greater; a tie keeps the left
 * (lower index). Per-chunk, per-thread and global merges all use that rule,
 * so the combined result equals the sequential reference scan (NaN included:
 * a leading NaN poisons exactly as in the reference, since x > NaN is false).
 *
 * Each OpenMP thread scans a contiguous span with 8-wide AVX-512 lanes
 * (4-wide AVX2 or scalar fallback). Lane k carries (max value, index of its
 * first max). Horizontal reduction: global lane max, then first (lowest)
 * lane holding it, which is the lowest index. */

#include <stdint.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>
#include <string.h>
#include <stdio.h>

/* ---- adaptive NUMA placement ----
 * The driver process's affinity mask is narrowed (main on CPU 0) while our
 * worker threads carry a wide mask and can run on any of the 192 cores.
 * The 3.5GB input array sits on some NUMA node; streaming it from other
 * nodes costs ~2x bandwidth. We read the page placement of the VMA that
 * holds the input from /proc/self/numa_maps (no privileged syscalls) and
 * pin each worker to a physical core of the right node(s) via the raw
 * sched_setaffinity syscall (the libc prototypes are hidden by
 * -D_POSIX_C_SOURCE). Result: data on one node -> all threads there;
 * interleaved data -> threads spread proportionally across nodes. */

#include <unistd.h>

#define AMAX_SYS_SCHED_SETAFFINITY 203
extern long syscall(long number, ...);
#define AMAX_MAXNODES 8

struct amax_aff {
  unsigned long b[16]; /* 1024 cpus */
};

typedef struct {
  int ncpus[AMAX_MAXNODES];
  int cpus[AMAX_MAXNODES][64];
  int plan_node[48];   /* node for worker t */
  int n_nodes;         /* 0 => no pinning plan */
} amax_plan;

static int amax_parse_cpu_list(const char *s, int out[64], int max) {
  int n = 0;
  const char *p = s;
  while (*p && n < max) {
    int lo = (int)strtol(p, (char **)&p, 10);
    int hi = lo;
    if (*p == '-') {
      p++;
      hi = (int)strtol(p, (char **)&p, 10);
    }
    for (int c = lo; c <= hi && n < max; ++c)
      out[n++] = c;
    if (*p == ',')
      p++;
    else
      break;
  }
  return n;
}

static void amax_load_node_cpus(amax_plan *pl) {
  pl->n_nodes = 0;
  for (int nd = 0; nd < AMAX_MAXNODES; ++nd) {
    char path[64];
    snprintf(path, sizeof path, "/sys/devices/system/node/node%d/cpulist", nd);
    FILE *f = fopen(path, "r");
    if (!f)
      continue;
    char buf[512];
    size_t r = fread(buf, 1, sizeof buf - 1, f);
    buf[r] = 0;
    fclose(f);
    /* physical cores first: the first contiguous run in the list */
    int n = amax_parse_cpu_list(buf, pl->cpus[nd], 64);
    int run = 0;
    for (int c = 0; c < n; ++c) {
      if (c > 0 && pl->cpus[nd][c] == pl->cpus[nd][c - 1] + 1) {
        pl->cpus[nd][n + run] = pl->cpus[nd][c];
        run++;
      } else
        break;
    }
    n += run; /* keep physical run + its SMT siblings */
    if (n > 1) {
      pl->ncpus[nd] = n;
      pl->n_nodes++;
    }
  }
}

/* Find the VMA holding a and sum per-node page counts. numa_maps lists
 * VMAs in ascending order: "startaddr(pages-hex, no 0x) ... N0=.. N1=..".
 * A VMA ends where the next one starts. Returns 1 on success. */
static int amax_find_placement(const void *a, int pages[AMAX_MAXNODES]) {
  for (int k = 0; k < AMAX_MAXNODES; ++k)
    pages[k] = 0;
  unsigned long a0 = (unsigned long)a;
  FILE *f = fopen("/proc/self/numa_maps", "r");
  if (!f)
    return 0;
  char line[1600];
  int found = 0;
  unsigned long prev_start = 0;
  int prev_pages[AMAX_MAXNODES];
  int have_prev = 0;
  while (fgets(line, sizeof line, f)) {
    char *sp = strchr(line, ' ');
    if (!sp)
      continue;
    *sp = 0;
    unsigned long start = 0;
    const char *p = line;
    while (*p) {
      if (*p >= '0' && *p <= '9')
        start = start * 16 + (unsigned long)(*p - '0');
      else if (*p >= 'a' && *p <= 'f')
        start = start * 16 + (unsigned long)(*p - 'a' + 10);
      else if (*p >= 'A' && *p <= 'F')
        start = start * 16 + (unsigned long)(*p - 'A' + 10);
      else
        break;
      p++;
    }
    *sp = ' ';
    if (p == line)
      continue;
    int cur_pages[AMAX_MAXNODES];
    for (int k = 0; k < AMAX_MAXNODES; ++k)
      cur_pages[k] = 0;
    for (int k = 0; k < AMAX_MAXNODES; ++k) {
      char tag[8];
      snprintf(tag, sizeof tag, "N%d=", k);
      char *q = strstr(sp + 1, tag);
      if (q)
        cur_pages[k] = (int)strtol(q + (int)strlen(tag), NULL, 10);
    }
    if (have_prev && a0 >= prev_start && a0 < start) {
      for (int k = 0; k < AMAX_MAXNODES; ++k)
        pages[k] = prev_pages[k];
      found = 1;
      break;
    }
    prev_start = start;
    for (int k = 0; k < AMAX_MAXNODES; ++k)
      prev_pages[k] = cur_pages[k];
    have_prev = 1;
  }
  if (!found && have_prev && a0 >= prev_start) {
    for (int k = 0; k < AMAX_MAXNODES; ++k)
      pages[k] = prev_pages[k];
    found = 1;
  }
  fclose(f);
  return found;
}

static void amax_make_plan(const void *a, int nworkers, amax_plan *pl) {
  int pages[AMAX_MAXNODES];
  if (!amax_find_placement(a, pages)) {
    pl->n_nodes = 0;
    return;
  }
  long total = 0;
  for (int k = 0; k < AMAX_MAXNODES; ++k)
    total += pages[k];
  if (total <= 0) {
    pl->n_nodes = 0;
    return;
  }
  /* a single node holding more than half the pages -> everyone there */
  int dom = -1;
  for (int k = 0; k < AMAX_MAXNODES; ++k)
    if ((long)pages[k] * 2 > total)
      dom = k;
  if (dom >= 0) {
    for (int t = 0; t < nworkers; ++t)
      pl->plan_node[t] = dom;
    pl->n_nodes = 1;
    return;
  }
  /* spread proportionally across nodes holding more than 5% of pages */
  int eligible = 0;
  int dom2 = 0;
  for (int k = 0; k < AMAX_MAXNODES; ++k) {
    if ((long)pages[k] * 20 > total)
      eligible++;
    if (pages[k] > pages[dom2])
      dom2 = k;
  }
  if (eligible == 0) {
    pl->n_nodes = 0;
    return;
  }
  int t = 0;
  for (int k = 0; k < AMAX_MAXNODES && t < nworkers; ++k) {
    if ((long)pages[k] * 20 <= total)
      continue;
    int share = (int)(((long)pages[k] * nworkers) / total);
    if (share < 1)
      share = 1;
    if (t + share > nworkers)
      share = nworkers - t;
    for (int j = 0; j < share && t < nworkers; ++j)
      pl->plan_node[t++] = k;
  }
  for (; t < nworkers; ++t)
    pl->plan_node[t] = dom2;
  pl->n_nodes = 1;
}

static void amax_pin_worker(int t, const amax_plan *pl) {
  if (pl->n_nodes <= 0)
    return;
  int nd = pl->plan_node[t];
  int ncpu = pl->ncpus[nd];
  if (nd < 0 || nd >= AMAX_MAXNODES || ncpu < 1)
    return;
  int cpu = pl->cpus[nd][t % ncpu];
  struct amax_aff aff;
  memset(&aff, 0, sizeof aff);
  aff.b[cpu >> 6] |= 1UL << (cpu & 63);
  syscall(AMAX_SYS_SCHED_SETAFFINITY, 0, (size_t)sizeof aff.b, &aff);
}



/* Merge of (value, first-index) pairs: larger value wins; on equal value the
 * smaller index wins. A pair is never NaN (NaN never wins a lane). */
static inline void pair_finish(const double *vals, const int64_t *idxs, int n,
                               double *fv, int64_t *fi) {
  *fv = vals[0];
  *fi = idxs[0];
  for (int k = 1; k < n; ++k) {
    if (vals[k] > *fv || (vals[k] == *fv && idxs[k] < *fi)) {
      *fv = vals[k];
      *fi = idxs[k];
    }
  }
}

/* ---- 8-wide (AVX-512) ---- */
#if defined(__AVX512F__)

static inline void chunk_scan8(const double *restrict a, int64_t start, int64_t end,
                               double *ov, int64_t *oi) {
  if (start >= end) { *ov = -HUGE_VAL; *oi = start; return; }
  double sv = a[start];
  int64_t si = start;
  int64_t i = start + 1;
  while (i < end && ((uintptr_t)(a + i) & 63)) {
    double t = a[i];
    if (t > sv) { sv = t; si = i; }
    ++i;
  }
  __m512d vv = _mm512_set1_pd(-HUGE_VAL);
  __m512d vi = _mm512_set_pd(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
  const __m512d zero_seven = vi;
  const int64_t n8 = (end - i) / 8;
  for (int64_t k = 0; k < n8; ++k, i += 8) {
    __builtin_prefetch(a + i + 512, 0, 3);
    __m512d xv = _mm512_load_pd(a + i);
    __m512d xi = _mm512_add_pd(_mm512_set1_pd((double)i), zero_seven);
    __mmask8 m = _mm512_cmp_pd_mask(xv, vv, _CMP_GT_OQ);
    vv = _mm512_mask_blend_pd(m, vv, xv);
    vi = _mm512_mask_blend_pd(m, vi, xi);
  }
  if (n8) {
    double vals[8];
    int64_t idxs[8];
    _mm512_store_pd(vals, vv);
    _mm512_store_si512(idxs, _mm512_cvtpd_epi64(vi));
    double fv;
    int64_t fi;
    pair_finish(vals, idxs, 8, &fv, &fi);
    if (fv > sv) { sv = fv; si = fi; }
  }
  for (; i < end; ++i) {
    double t = a[i];
    if (t > sv) { sv = t; si = i; }
  }
  *ov = sv;
  *oi = si;
}

#endif /* __AVX512F__ */

/* ---- 4-wide (AVX2) ---- */
#if defined(__AVX2__)

static inline void amax_vec4_merge(__m256d *vv, __m256d *vi, __m256d xv, __m256d xi) {
  __m256d m = _mm256_cmp_pd(xv, *vv, _CMP_GT_OQ);
  *vv = _mm256_blendv_pd(*vv, xv, m);
  *vi = _mm256_blendv_pd(*vi, xi, m);
}

static inline void chunk_scan4(const double *restrict a, int64_t start, int64_t end,
                               double *ov, int64_t *oi) {
  if (start >= end) { *ov = -HUGE_VAL; *oi = start; return; }
  double sv = a[start];
  int64_t si = start;
  int64_t i = start + 1;
  while (i < end && ((uintptr_t)(a + i) & 31)) {
    double t = a[i];
    if (t > sv) { sv = t; si = i; }
    ++i;
  }
  __m256d vv = _mm256_set1_pd(-HUGE_VAL);
  __m256d vi = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
  const __m256d zero_three = vi;
  const int64_t n4 = (end - i) / 4;
  for (int64_t k = 0; k < n4; ++k, i += 4) {
    __builtin_prefetch(a + i + 512, 0, 3);
    __m256d xv = _mm256_load_pd(a + i);
    __m256d xi = _mm256_add_pd(_mm256_set1_pd((double)i), zero_three);
    amax_vec4_merge(&vv, &vi, xv, xi);
  }
  if (n4) {
    double vals[4];
    int64_t idxs[4];
    _mm256_store_pd(vals, vv);
    _mm256_store_si256((void *)idxs, _mm256_cvtpd_epi64(vi));
    double fv;
    int64_t fi;
    pair_finish(vals, idxs, 4, &fv, &fi);
    if (fv > sv) { sv = fv; si = fi; }
  }
  for (; i < end; ++i) {
    double t = a[i];
    if (t > sv) { sv = t; si = i; }
  }
  *ov = sv;
  *oi = si;
}

#endif /* __AVX2__ */

/* ---- scalar ---- */
static inline void chunk_scan1(const double *restrict a, int64_t start, int64_t end,
                               double *ov, int64_t *oi) {
  if (start >= end) { *ov = -HUGE_VAL; *oi = start; return; }
  double sv = a[start];
  int64_t si = start;
  for (int64_t i = start + 1; i < end; ++i) {
    double t = a[i];
    if (t > sv) { sv = t; si = i; }
  }
  *ov = sv;
  *oi = si;
}

void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D) {
  if (LEN_1D <= 0) {
    out_value[0] = a[0];
    out_index[0] = 0;
    return;
  }
  const int64_t T = (int64_t)omp_get_max_threads();
  double *pv = (double *)malloc((size_t)T * (sizeof(double) + sizeof(int64_t)));
  int64_t *pi = (int64_t *)(pv + T);
  static amax_plan cached;
  static const void *cached_a = NULL;
  static int64_t cached_len = -1;
  if (a != cached_a || LEN_1D != cached_len) {
    amax_load_node_cpus(&cached);
    amax_make_plan(a, (int)T, &cached);
    cached_a = a;
    cached_len = LEN_1D;
  }
  const amax_plan *pl = &cached;
#if defined(__AVX512F__)
  const int mode = __builtin_cpu_supports("avx512f") ? 2 :
                   (__builtin_cpu_supports("avx2") ? 1 : 0);
#elif defined(__AVX2__)
  const int mode = __builtin_cpu_supports("avx2") ? 1 : 0;
#endif
  #pragma omp parallel
  {
    const int64_t t = (int64_t)omp_get_thread_num();
    amax_pin_worker((int)t, pl);
    const int64_t nt = (int64_t)omp_get_num_threads();
    const int64_t base = (LEN_1D * t) / nt;
    const int64_t end = (LEN_1D * (t + 1)) / nt;
    double bv;
    int64_t bi;
#if defined(__AVX512F__)
    if (mode == 2)
      chunk_scan8(a, base, end, &bv, &bi);
    else
#endif
#if defined(__AVX2__)
    if (mode == 1)
      chunk_scan4(a, base, end, &bv, &bi);
    else
#endif
      chunk_scan1(a, base, end, &bv, &bi);
    pv[t] = bv;
    pi[t] = bi;
  }
  int64_t first = 0;
  while (first < T && (LEN_1D * (first + 1)) / T == 0)
    ++first;
  double bv = pv[first];
  int64_t bi = pi[first];
  for (int64_t t = 0; t < T; ++t) {
    if (t == first)
      continue;
    if (pv[t] > bv) {
      bv = pv[t];
      bi = pi[t];
    }
  }
  free(pv);
  out_value[0] = bv;
  out_index[0] = bi;
}
