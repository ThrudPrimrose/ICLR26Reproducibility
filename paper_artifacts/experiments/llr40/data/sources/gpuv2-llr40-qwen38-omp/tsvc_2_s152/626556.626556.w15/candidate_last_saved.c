#include <stdint.h>
#include <stdio.h>
#include <omp.h>

static int64_t g_calls = 0;
static double now_ns(void) { return (double)(omp_get_wtime() * 1e9); }

void tsvc_2_s152_fp64(double *restrict a, double *restrict b, const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
    g_calls++;
    long long n = (long long)LEN_1D;
    double t0 = now_ns();

    int on_dev = 0, maxt_dev = 0, nteams = 0;
    #pragma omp target map(from: on_dev, maxt_dev, nteams)
    #pragma omp teams
    {
        on_dev = !omp_is_initial_device();
        maxt_dev = omp_get_max_threads();
        nteams = omp_get_num_teams();
    }

    double t1 = now_ns();
    #pragma omp target map(tofrom: a[0:n], b[0:n]) map(to: c[0:n], d[0:n], e[0:n])
    {
        #pragma omp teams distribute parallel for
        for (long long i = 0; i < n; ++i) {
            double bi = d[i] * e[i];
            b[i] = bi;
            a[i] += bi * c[i];
        }
    }
    double t2 = now_ns();

    {
        static FILE *f = NULL;
        if (!f) { f = fopen("/shared/agent-15/calls2.log", "a"); if (f) setvbuf(f, NULL, _IONBF, 0); }
        if (f) fprintf(f, "call=%lld on_dev=%d maxt_dev=%d teams=%d host_maxt=%d t_enter=%.0f t_region=%.0f t_total=%.0f\n",
                       (long long)g_calls, on_dev, maxt_dev, nteams, omp_get_max_threads(), t1-t0, t2-t1, now_ns()-t0);
    }
}
