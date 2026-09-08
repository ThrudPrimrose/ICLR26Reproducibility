#include <stdint.h>
#include <stdio.h>
#include <omp.h>

/* ts119: aa[i][j] = aa[i-1][j-1] + bb[i][j] for 1<=i<N, 1<=j<N.
 * Each anti-diagonal d = j - i is an independent sequential scan;
 * boundaries (row 0 / col 0) are never written.
 * Host path for small N (avoids offload round-trip cost),
 * GPU target path for large N. */

static int g_printed = 0;

static inline void run_chain(double *restrict aa, const double *restrict bb,
                             const int64_t N, const int64_t d) {
    int64_t i0 = 1 - d;
    if (i0 < 1) i0 = 1;
    int64_t i1 = N - 1 - d;
    if (i1 > N - 1) i1 = N - 1;
    int64_t j0 = i0 + d;
    double acc = aa[(i0 - 1) * N + (j0 - 1)];
    for (int64_t i = i0, j = j0; i <= i1; ++i, ++j) {
        acc += bb[i * N + j];
        aa[i * N + j] = acc;
    }
}

static void host_serial(double *restrict aa, const double *restrict bb, const int64_t N) {
    for (int64_t i = 1; i < N; ++i) {
        for (int64_t j = 1; j < N; ++j) {
            aa[i * N + j] = aa[(i - 1) * N + (j - 1)] + bb[i * N + j];
        }
    }
}

static void gpu_run(double *restrict aa, const double *restrict bb, const int64_t N) {
    const int64_t nn = N * N;
    const int64_t nchains = 2 * N - 3;
    int64_t nteams = (nchains + 63) / 64;
    if (nteams < 1) nteams = 1;
    if (nteams > 8192) nteams = 8192;
    #pragma omp target map(tofrom: aa[0:nn]) map(to: bb[0:nn])
    {
        #pragma omp teams num_teams(nteams)
        {
            const int64_t lane = omp_get_thread_num();
            const int64_t nl = omp_get_num_threads();
            const int64_t dlo = 2 - N;
            for (int64_t k = lane; k < nchains; k += nl) {
                run_chain(aa, bb, N, dlo + k);
            }
        }
    }
}

void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    const int64_t N = LEN_2D;
    if (!g_printed) {
        g_printed = 1;
        printf("PROBE N=%lld\n", (long long)N);
        fflush(stdout);
    }
    if (N < 2) return;
    if (N < 1024) {
        host_serial(aa, bb, N);
    } else {
        gpu_run(aa, bb, N);
    }
}
