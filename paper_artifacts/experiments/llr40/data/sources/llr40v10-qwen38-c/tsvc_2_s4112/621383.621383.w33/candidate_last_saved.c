/* Optimized tsvc_2 s4112: a[i] += b[ip[i]] * 2.0  (random gather, fp64)
 *
 * Strategy:
 *  - Embarrassingly parallel over i; pin OpenMP threads to the NUMA node
 *    that owns array b (detected via libnuma numa_pages_origin, with a
 *    latency-probe fallback) so all random gathers are local.
 *  - Use the physical core count of that node (SMT siblings do not help a
 *    memory-latency-bound gather; mixing local+remote threads badly
 *    contends the inter-socket links).
 *  - Intersect the node's CPUs with the process affinity mask so we never
 *    fight a harness-imposed taskset; if the intersection is tiny, fall
 *    back to the process mask as a whole.
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <sched.h>
#include <dlfcn.h>
#include <omp.h>

#define MAXCPUS 512

static int g_cpu[MAXCPUS]; /* one physical core (a logical cpu id) per entry */
static int g_ncpu = 0;
static int g_built_node = -1;

/* parse a kernel cpulist string like "0-23,96-119" into an array */
static int parse_cpulist(const char *path, int *cpus, int cap) {
  FILE *f = fopen(path, "r");
  if (!f) return -1;
  int n = 0;
  for (int c = fgetc(f); c != EOF && n < cap; c = fgetc(f)) {
    if (isdigit((unsigned char)c)) {
      int a = c - '0';
      int c2 = fgetc(f);
      if (c2 == '-') {
        int b = 0;
        c2 = fgetc(f);
        while (c2 != EOF && isdigit((unsigned char)c2)) { b = b * 10 + (c2 - '0'); c2 = fgetc(f); }
        for (int i = a; i <= b && n < cap; i++) cpus[n++] = i;
      } else {
        cpus[n++] = a;
      }
    }
  }
  fclose(f);
  return n;
}

/* build g_cpu[]: one representative logical cpu per physical core of the node */
static int build_node_cpus(int node) {
  char path[256];
  snprintf(path, sizeof path, "/sys/devices/system/node/node%d/cpulist", node);
  int log[MAXCPUS];
  int nlog = parse_cpulist(path, log, MAXCPUS);
  if (nlog <= 0) return -1;
  g_ncpu = 0;
  /* try to dedupe SMT siblings via topology/core_id */
  int cores[MAXCPUS], ncores = 0, have_cores = 1;
  for (int i = 0; i < nlog && g_ncpu < MAXCPUS; i++) {
    char p2[288];
    int cid = -1;
    snprintf(p2, sizeof p2, "/sys/devices/system/cpu/cpu%d/topology/core_id", log[i]);
    FILE *f = fopen(p2, "r");
    if (f) { if (fscanf(f, "%d", &cid) != 1) { fclose(f); have_cores = 0; cid = -1; } else fclose(f); }
    else have_cores = 0;
    if (!have_cores) { g_cpu[g_ncpu++] = log[i]; continue; }
    int dup = 0;
    for (int j = 0; j < ncores; j++) if (cores[j] == cid) { dup = 1; break; }
    if (!dup) { cores[ncores++] = cid; g_cpu[g_ncpu++] = log[i]; }
  }
  return g_ncpu;
}

/* node that owns b: libnuma numa_pages_origin; fallback = latency probe */
static int detect_node(const double *b, int64_t N) {
  typedef int (*pages_origin_t)(const void *, int *);
  static pages_origin_t f = (pages_origin_t)0;
  static int tried = 0;
  if (!tried) {
    tried = 1;
    void *h = dlopen("libnuma.so.1", RTLD_NOW | RTLD_LOCAL);
    if (h) f = (pages_origin_t)dlsym(h, "numa_pages_origin");
  }
  if (f) {
    int node = -1;
    if (f((const void *)b, &node) >= 0 && node >= 0) return node;
  }
  /* fallback: measure random-gather latency per node, keep the best */
  char dirp[256];
  int nodes[MAXCPUS], nn = 0;
  for (int n = 0; n < 64 && nn < MAXCPUS; n++) {
    snprintf(dirp, sizeof dirp, "/sys/devices/system/node/node%d/cpulist", n);
    if (parse_cpulist(dirp, (int[8]){0}, 8) > 0) nodes[nn++] = n;
  }
  if (nn <= 0) return 0;
  double gbest = 1e300;
  int best = nodes[0];
  for (int k = 0; k < nn; k++) {
    char path[256];
    snprintf(path, sizeof path, "/sys/devices/system/node/node%d/cpulist", nodes[k]);
    int log[16];
    int nlog = parse_cpulist(path, log, 16);
    if (nlog < 1) continue;
    double acc = 0.0;
    double t0 = omp_get_wtime();
    #pragma omp parallel num_threads(1)
    {
      cpu_set_t s; CPU_ZERO(&s); CPU_SET(log[0], &s);
      sched_setaffinity(0, sizeof s, &s);
      uint64_t h = 0x9e3779b97f4a7c15ull;
      for (int i = 0; i < 20000; i++) {
        h = h * 6364136223846793005ull + 1442695040888963407ull;
        acc += b[(h >> 11) % (uint64_t)N];
      }
    }
    double dt = omp_get_wtime() - t0;
    if (acc == -42.0) dt = 1e300; /* anti-DCE */
    if (dt < gbest) { gbest = dt; best = nodes[k]; }
  }
  return best;
}

void tsvc_2_s4112_fp64(double *restrict a, const double *restrict b,
                       const int32_t *restrict ip, const int64_t LEN_1D) {
  if (LEN_1D <= 0) return;

  int node = detect_node(b, LEN_1D);
  if (node != g_built_node) {
    if (build_node_cpus(node) > 0) g_built_node = node;
  }
  if (g_ncpu <= 0) { /* no sysfs info: plain parallel */ }

  /* intersect the node's physical cores with the process affinity mask */
  cpu_set_t pmask;
  CPU_ZERO(&pmask);
  sched_getaffinity(0, sizeof pmask, &pmask);
  int loc[MAXCPUS], nloc = 0;
  for (int i = 0; i < g_ncpu; i++)
    if (CPU_ISSET(g_cpu[i], &pmask)) loc[nloc++] = g_cpu[i];

  int T;
  const int *target;
  if (nloc >= 4) {
    T = nloc;
    target = loc;
  } else {
    /* restricted process mask: spread over what we are allowed to use */
    nloc = 0;
    for (int c = 0; c < (int)sizeof(pmask) * 8 && nloc < MAXCPUS; c++)
      if (CPU_ISSET(c, &pmask)) loc[nloc++] = c;
    if (nloc <= 0) nloc = 1;
    T = nloc;
    target = loc;
  }
  if (T > 256) T = 256;

  #pragma omp parallel num_threads(T)
  {
    int t = omp_get_thread_num();
    if (t < T) {
      cpu_set_t s;
      CPU_ZERO(&s);
      CPU_SET(target[t], &s);
      sched_setaffinity(0, sizeof s, &s);
    }
    int64_t chunk = LEN_1D / T;
    if (chunk == 0) chunk = 1;
    int64_t lo = (int64_t)t * chunk;
    int64_t hi = lo + chunk;
    if (t == T - 1) hi = LEN_1D;
    for (int64_t i = lo; i < hi; ++i)
      a[i] += b[ip[i]] * 2.0;
  }
}
