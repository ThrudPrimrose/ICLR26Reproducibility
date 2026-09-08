#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void scan_affine_decay_fp64(double *y, double *c, double *x, int64_t N,
                            uint8_t *ws, int64_t ws_bytes)
{
    (void)ws; (void)ws_bytes;
    static int done = 0;
    if (!done) {
        done = 1;
        printf("DIAG max_threads=%d num_procs=%d omp_num_threads=%d\n",
               omp_get_max_threads(), omp_get_num_procs(), omp_get_num_threads());
        FILE *st = fopen("/proc/self/status", "r");
        if (st) {
            char line[512];
            while (fgets(line, sizeof line, st)) {
                if (!strncmp(line, "Cpus_allowed_list:", 18)) {
                    int cnt = 0;
                    for (const char *p = line + 18; *p; p++) if (*p == '-' || *p == ',') cnt++;
                    cnt = 0;
                    for (const char *p = line + 18; *p; p++) if (*p >= '0' && *p <= '9') cnt++;
                    printf("DIAG cpulist=[%s] ndigits=%d\n", line + 18, cnt);
                }
            }
            fclose(st);
        }
        (void)!system("grep -m1 'model name' /proc/cpuinfo");
        (void)!system("lscpu | grep -E 'NUMA node\\(s\\)|L3 cache|Socket'");
        (void)!system("ls /sys/class/kfd/kfd/topology/nodes 2>/dev/null | wc -l");
        (void)!system("cat /proc/cpuinfo | grep -m1 MHz");
        printf("DIAG HIP_VISIBLE_DEVICES=%s OMP_TARGET_OFFLOAD=%s\n",
               getenv("HIP_VISIBLE_DEVICES") ? getenv("HIP_VISIBLE_DEVICES") : "(null)",
               getenv("OMP_TARGET_OFFLOAD") ? getenv("OMP_TARGET_OFFLOAD") : "(null)");
        const int64_t M = (1 << 28) / 8; /* 256 MB */
        double *hbuf = (double *)malloc(M * 8);
        for (int64_t i = 0; i < M; i++) hbuf[i] = (double)i;
        double *dbuf = (double *)omp_target_alloc(M * 8, 0);
        /* warm */
        #pragma omp target map(to: dbuf[0:M]) map(from: hbuf[0:M])
        memcpy(dbuf, hbuf, M * 8);
        double t0 = omp_get_wtime();
        #pragma omp target map(to: dbuf[0:M]) map(from: hbuf[0:M])
        memcpy(dbuf, hbuf, M * 8);
        double t1 = omp_get_wtime();
        printf("DIAG H2D %.1f GB/s (256MB)\n", (M * 8) / 1e9 / (t1 - t0));
        #pragma omp target data map(to: hbuf[0:M])
        {
            t0 = omp_get_wtime();
            #pragma omp target update from(hbuf[0:M])
            t1 = omp_get_wtime();
        }
        printf("DIAG D2H %.1f GB/s (256MB)\n", (M * 8) / 1e9 / (t1 - t0));
        omp_target_free(dbuf, 0);
        free(hbuf);
        fflush(stdout);
    }
    if (N <= 0) return;
    y[0] = x[0];
    for (int64_t i = 1; i < N; i++) y[i] = fma(c[i], y[i - 1], x[i]);
}
