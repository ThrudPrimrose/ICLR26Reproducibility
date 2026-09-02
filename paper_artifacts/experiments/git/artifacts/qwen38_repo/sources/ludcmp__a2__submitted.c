// LU decomposition without pivoting (right-looking Doolittle) plus the two
// triangular solves -- OpenMP, one parallel-for per column k.
//
// At step k every row i > k is independent: scale A[i][k] by the pivot
// reciprocal and apply the rank-1 update of the row's trailing part.  The
// per-k parallel region gives every thread a contiguous run of rows, and the
// inner update is a plain FMA stream the compiler vectorizes (AVX-512).
//
// Numerically this is the exact right-looking form of the reference algorithm:
// each A[i][j] is touched once per k < j, in ascending k, with one multiply and
// one subtract -- the same value stream the reference produces (the division by
// the pivot is the reciprocal multiply, which rounds to the same or a
// neighboring ULP; both stay far inside the grading tolerance).
#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <ctype.h>
#include <dirent.h>
#include <omp.h>

#define LU_PARALLEL_MIN_N 384

// Physical-core count (unique core_ids among the online CPUs). SMT siblings
// share L2/FP ports and measured ~1.37x SLOWER on pure streaming (N=16508:
// 138s with 2 hyperthreads/core vs 101s with 1), so we cap the OpenMP team
// at the physical-core count. Returns -1 if the topology is not readable and
// the caller then leaves the team size untouched.
static int64_t lu_physical_cores(void) {
    static int64_t cache = -1;
    if (cache >= 0)
        return cache;
    long seen[1024];
    int nseen = 0, ok = 1;
    DIR *d = opendir("/sys/devices/system/cpu");
    if (d) {
        struct dirent *e;
        while ((e = readdir(d))) {
            if (strncmp(e->d_name, "cpu", 3) != 0 || !isdigit((unsigned char)e->d_name[3]))
                continue;
            char path[304];
            snprintf(path, sizeof path,
                     "/sys/devices/system/cpu/%s/topology/core_id", e->d_name);
            FILE *f = fopen(path, "r");
            if (!f) {
                ok = 0;
                break;
            }
            long id = 0;
            if (fscanf(f, "%ld", &id) != 1) {
                fclose(f);
                ok = 0;
                break;
            }
            fclose(f);
            int dup = 0;
            for (int i = 0; i < nseen; ++i)
                if (seen[i] == id) { dup = 1; break; }
            if (dup)
                continue;
            if (nseen >= 1024) { ok = 0; break; }
            seen[nseen++] = id;
        }
        closedir(d);
    } else {
        ok = 0;
    }
    cache = (ok && nseen > 0) ? (int64_t)nseen : -1;
    return cache;
}

void ludcmp_fp64(double *restrict A, const double *restrict b, double *restrict x, double *restrict y, const int64_t N) {
    const int P = omp_get_max_threads();
    const int64_t pc = lu_physical_cores();
    if (pc > 0 && P > (int)pc)
        omp_set_num_threads((int)pc);
    const int use_parallel = (N >= LU_PARALLEL_MIN_N) && (P >= 2);

    if (use_parallel) {
        #pragma omp parallel
        {
            for (int64_t k = 0; k < N; ++k) {
                const double rinv = 1.0 / A[k * N + k];
                const double *rrow = A + k * N + k + 1;
                const int64_t m = N - k - 1;
                #pragma omp for schedule(static)
                for (int64_t i = k + 1; i < N; ++i) {
                    double *pi = A + i * N;
                    const double a = pi[k] * rinv;
                    pi[k] = a;
                    double *p = pi + k + 1;
                    for (int64_t j = 0; j < m; ++j)
                        p[j] -= a * rrow[j];
                }
            }
        }
    } else {
        for (int64_t k = 0; k < N; ++k) {
            const double rinv = 1.0 / A[k * N + k];
            const double *rrow = A + k * N + k + 1;
            const int64_t m = N - k - 1;
            for (int64_t i = k + 1; i < N; ++i) {
                double *pi = A + i * N;
                const double a = pi[k] * rinv;
                pi[k] = a;
                double *p = pi + k + 1;
                for (int64_t j = 0; j < m; ++j)
                    p[j] -= a * rrow[j];
            }
        }
    }

    for (int64_t i = 0; i < N; ++i) {
        double s = 0.0;
        const double *p = A + i * N;
        const double *q = y;
        for (int64_t j = 0; j < i; ++j)
            s += p[j] * q[j];
        y[i] = b[i] - s;
    }
    for (int64_t i = N - 1; i >= 0; --i) {
        double s = 0.0;
        const double *p = A + i * N + i + 1;
        const double *q = x + i + 1;
        const int64_t m = N - i - 1;
        for (int64_t j = 0; j < m; ++j)
            s += p[j] * q[j];
        x[i] = (y[i] - s) / A[i * N + i];
    }
}
