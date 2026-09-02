/* wf_north_west: exact tile-wavefront with ROW-MAJOR tile traversal.
 *
 * Correctness: element (i,j) = a[i][j] + a[i-1][j] + a[i][j-1]. Inside a tile
 * a plain row-major sweep computes every (i-1,j) (previous row) and (i,j-1)
 * (previous column) before (i,j); between tiles, the classic block
 * anti-diagonal phase K = bi+bj guarantees the up- and left-tiles (phase K-1)
 * are complete. Bit-exact, and - unlike a diagonal walk - the inner sweep is
 * contiguous, so no L1/L2/L3 set aliasing for any N (in particular N = 1 mod 64
 * where the diagonal stride (N-1)*8 is a multiple of the L1 size).
 *
 * Threading: if the process is pinned to <=2 CPUs a single serial pass runs
 * (no OMP). Otherwise one OMP thread per CPU of the NUMA node that holds `a`
 * (detected via /proc/self/maps + /proc/self/numa_maps, with fallbacks), and
 * the caller's affinity is restored afterwards. Detection is cached across
 * calls. Degrades gracefully to <=12 unpinned threads.
 *
 * Deliberately free of GNU-specific declarations (compiles under -std=c23 even
 * when the TU's feature macros were fixed before this file): affinity syscalls
 * are declared by hand, the CPU mask is a plain bit array.
 */
#include <stdint.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern int sched_setaffinity(int pid, size_t cpusetsize, const unsigned long *mask);
extern int sched_getaffinity(int pid, size_t cpusetsize, unsigned long *mask);

#define MAXCPU 1024
#define PIN_WORDS 16 /* 1024 CPUs, layout = kernel cpumask */

static int g_cpus[MAXCPU];
static int g_ncpus = 0;
static int g_allowed_n = -1; /* cached allowed-CPU count (-1 = not probed) */

static int parse_cpulist(const char *s, int *arr, int cap) {
    int n = 0;
    while (*s && n < cap) {
        const char *seg = s;
        while (*s && *s != ',') ++s;
        char buf[32];
        int len = (int)(s - seg);
        if (len >= (int)sizeof buf) len = (int)sizeof buf - 1;
        memcpy(buf, seg, len);
        buf[len] = 0;
        const char *dash = strchr(buf, '-');
        int lo = atoi(buf);
        int hi = dash ? atoi(dash + 1) : lo;
        if (hi < lo) { int t = lo; lo = hi; hi = t; }
        for (int c = lo; c <= hi && n < cap; ++c) arr[n++] = c;
        if (*s == ',') ++s;
    }
    return n;
}

static void scan_nfields(const char *p, unsigned long long *anontotal,
                         int *best, unsigned long long *bestk) {
    while (*p) {
        if (*p == 'N' && p[1] >= '0' && p[1] <= '9') {
            int node = 0;
            while (p[1] >= '0' && p[1] <= '9') { node = node * 10 + (p[1] - '0'); ++p; }
            if (*p == '=') {
                unsigned long long k = strtoull(p + 1, NULL, 10);
                if (node < 64) anontotal[node] += k;
                if (k > *bestk) { *bestk = k; *best = node; }
            }
        }
        ++p;
    }
}

static int detect_node(const double *a) {
    unsigned long long target_start = 0;
    FILE *m = fopen("/proc/self/maps", "r");
    if (m) {
        char line[512];
        while (fgets(line, sizeof line, m)) {
            char *dash = strchr(line, '-');
            if (!dash) continue;
            unsigned long long st = strtoull(line, NULL, 16);
            unsigned long long en = strtoull(dash + 1, NULL, 16);
            if ((unsigned long long)a >= st && (unsigned long long)a < en) { target_start = st; break; }
        }
        fclose(m);
    }
    if (target_start == 0) return -1;
    int best = -1;
    unsigned long long bestk = 0;
    unsigned long long anontotal[64];
    memset(anontotal, 0, sizeof anontotal);
    FILE *f = fopen("/proc/self/numa_maps", "r");
    if (!f) return -1;
    char line[4096];
    while (fgets(line, sizeof line, f)) {
        char *sp = strchr(line, ' ');
        if (!sp) continue;
        *sp = 0;
        unsigned long long st = strtoull(line, NULL, 16);
        if (st != target_start) {
            if (strstr(sp, "anon=")) {
                int b2 = -1;
                unsigned long long k2 = 0;
                scan_nfields(sp, anontotal, &b2, &k2);
            }
            continue;
        }
        scan_nfields(sp, anontotal, &best, &bestk);
        break;
    }
    fclose(f);
    if (best >= 0) return best;
    unsigned long long am = 0;
    for (int k = 0; k < 64; ++k)
        if (anontotal[k] > am) { am = anontotal[k]; best = k; }
    return am > 0 ? best : -1;
}

static int current_cpu(void) {
    FILE *f = fopen("/proc/self/stat", "r");
    if (!f) return -1;
    char buf[1024];
    size_t r = fread(buf, 1, sizeof buf - 1, f);
    buf[r] = 0;
    fclose(f);
    char *cp = strrchr(buf, ')');
    if (!cp) return -1;
    int tok = 0;
    long val = -1;
    for (char *p = cp + 1;;) {
        while (*p == ' ') ++p;
        if (!*p) break;
        char *e = p;
        while (*e && *e != ' ') ++e;
        *e = 0;
        ++tok;
        if (tok == 36) val = atol(p);
        p = e + 1;
    }
    return (int)val;
}

static int cpu_node(int cpu) {
    for (int k = 0; k < 16; ++k) {
        char path[160];
        snprintf(path, sizeof path, "/sys/devices/system/node/node%d/cpulist", k);
        FILE *g = fopen(path, "r");
        if (!g) continue;
        char buf[8192];
        size_t r = fread(buf, 1, sizeof buf - 1, g);
        buf[r] = 0;
        fclose(g);
        int cpus[MAXCPU];
        int n = parse_cpulist(buf, cpus, MAXCPU);
        for (int i = 0; i < n; ++i)
            if (cpus[i] == cpu) return k;
    }
    return -1;
}

static int node_cpus_intersect(int node, const int *allowed, int nallowed, int *out, int cap) {
    char path[160];
    snprintf(path, sizeof path, "/sys/devices/system/node/node%d/cpulist", node);
    FILE *g = fopen(path, "r");
    if (!g) return 0;
    char buf[8192];
    size_t r = fread(buf, 1, sizeof buf - 1, g);
    buf[r] = 0;
    fclose(g);
    int nodecpus[MAXCPU];
    int nn = parse_cpulist(buf, nodecpus, MAXCPU);
    int n = 0;
    for (int i = 0; i < nn && n < cap; ++i)
        for (int j = 0; j < nallowed; ++j)
            if (nodecpus[i] == allowed[j]) { out[n++] = nodecpus[i]; break; }
    return n;
}

/* Row-major sweep of one tile. Exact: (i-1,j) is in the previous row,
 * (i,j-1) in the previous column; tile entry values come from phase K-1. */
static void tile_sweep(double *restrict a, const int64_t N,
                       const int64_t i0, const int64_t i1,
                       const int64_t j0, const int64_t j1) {
    for (int64_t i = i0; i <= i1; ++i) {
        int64_t p = i * N + j0;
        for (int64_t j = j0; j <= j1; ++j, ++p)
            a[p] = a[p] + a[p - N] + a[p - 1];
    }
}

typedef struct { unsigned long w[PIN_WORDS]; } pinmask_t;
static void pin_this_thread(int tid) {
    if (g_ncpus <= 0) return;
    pinmask_t cs;
    for (int i = 0; i < PIN_WORDS; ++i) cs.w[i] = 0;
    int c = g_cpus[tid % g_ncpus];
    cs.w[c >> 6] = 1ULL << (c & 63);
    sched_setaffinity(0, sizeof cs, cs.w);
}

static void run_parallel(double *restrict a, const int64_t N, int nt) {
    if (nt < 1) nt = 1;
    const int64_t T = 128;
    const int64_t nb = (N - 1 + T - 1) / T;
    pinmask_t oldmask;
    int have_mask = (g_ncpus > 0 &&
                     sched_getaffinity(0, sizeof oldmask, oldmask.w) == 0);
#pragma omp parallel num_threads(nt)
    {
        pin_this_thread(omp_get_thread_num());
        for (int64_t K = 0; K < 2 * nb - 1; ++K) {
            const int64_t bmin = K > nb - 1 ? K - (nb - 1) : 0;
            const int64_t bmax = K < nb - 1 ? K : nb - 1;
#pragma omp for
            for (int64_t bi = bmin; bi <= bmax; ++bi) {
                const int64_t bj = K - bi;
                const int64_t i0 = 1 + bi * T;
                const int64_t i1 = i0 + T - 1 < N ? i0 + T - 1 : N - 1;
                const int64_t j0 = 1 + bj * T;
                const int64_t j1 = j0 + T - 1 < N ? j0 + T - 1 : N - 1;
                tile_sweep(a, N, i0, i1, j0, j1);
            }
        }
    }
    if (have_mask) sched_setaffinity(0, sizeof oldmask, oldmask.w);
}

static void run_serial(double *restrict a, const int64_t N) {
    for (int64_t i = 1; i < N; ++i) {
        int64_t p = i * N + 1;
        for (int64_t j = 1; j < N; ++j, ++p)
            a[p] = a[p] + a[p - N] + a[p - 1];
    }
}

void wf_north_west_fp64(double *restrict a, const int64_t N) {
    if (N <= 2) {
        if (N == 2) a[N + 1] = a[N + 1] + a[1] + a[N];
        return;
    }
    if (N <= 512) {
        run_serial(a, N);
        return;
    }

    if (g_allowed_n < 0) {
        int allowed[MAXCPU];
        int nallowed = 0;
        FILE *f = fopen("/proc/self/status", "r");
        if (f) {
            char line[8192];
            while (fgets(line, sizeof line, f)) {
                if (strncmp(line, "Cpus_allowed_list:", 18) == 0) {
                    nallowed = parse_cpulist(line + 18, allowed, MAXCPU);
                    break;
                }
            }
            fclose(f);
        }
        g_allowed_n = nallowed;
        if (nallowed > 2) {
            int node = detect_node(a);
            if (node < 0) node = cpu_node(current_cpu());
            if (node >= 0) g_ncpus = node_cpus_intersect(node, allowed, nallowed, g_cpus, MAXCPU);
        }
    }

    int nt = omp_get_max_threads();
    int cap = g_ncpus > 0 ? g_ncpus : 12;
    if (nt > cap) nt = cap;
    run_parallel(a, N, nt);
}
