/* probe v2: NUMA placement + bound vs spread bandwidth. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sched.h>
#include <omp.h>
#include <immintrin.h>

static double now_ns(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return (double)ts.tv_sec*1e9+ts.tv_nsec; }

/* find the VMA containing `target` in /proc/self/numa_maps; print node counts. */
static void numa_report(const void *target, const char *tag)
{
    FILE *f = fopen("/proc/self/numa_maps", "r");
    if (!f) { fprintf(stdout, "PROBE numa_maps: (none)\n"); return; }
    char line[512];
    unsigned long prev_start = 0;
    char prev_line[512] = {0};
    int found = 0;
    while (fgets(line, sizeof line, f)) {
        unsigned long start = 0;
        if (sscanf(line, "%lx", &start) != 1) continue;
        if (start > (unsigned long)target) {
            if (prev_start <= (unsigned long)target) { found = 1; break; }
            /* target is before the first VMA we've seen -- unlikely */
            break;
        }
        prev_start = start;
        strcpy(prev_line, line);
    }
    fclose(f);
    if (found) fprintf(stdout, "PROBE numa %s: %s", tag, prev_line);
    else fprintf(stdout, "PROBE numa %s: (not found)\n", tag);
}

static void parse_cpulist(const char *text, cpu_set_t *set)
{
    CPU_ZERO(set);
    for (const char *p = text; *p;) {
        char *end = NULL;
        long a = strtol(p, &end, 10);
        if (end == p) break;
        if (*end == '-') {
            p = end + 1;
            long b = strtol(p, &end, 10);
            for (long c = a; c <= b; ++c) CPU_SET((int)c, set);
        } else {
            CPU_SET((int)a, set);
        }
        if (*end == ',') p = end + 1; else break;
    }
}

static void read_bw(const double *a, const double *b, int64_t n, int nt, const cpu_set_t *bind_set, double *t_ms, double *bw)
{
    double acc = 0.0;
    double t0 = now_ns();
    #pragma omp parallel for num_threads(nt) schedule(static,1) reduction(+:acc)
    for (int64_t i = 0; i < n; i += 8) {
        if (bind_set) sched_setaffinity(0, sizeof(cpu_set_t), bind_set);
        __m512d v = _mm512_loadu_pd(a + i);
        __m512d w = _mm512_loadu_pd(b + i);
        acc += _mm512_reduce_add_pd(_mm512_add_pd(v, w));
    }
    double t1 = now_ns();
    *t_ms = (t1-t0)*1e-3;
    *bw = 2.0*8.0*(double)n/(t1-t0);
}

int main_probe_placeholder(void){return 0;}
