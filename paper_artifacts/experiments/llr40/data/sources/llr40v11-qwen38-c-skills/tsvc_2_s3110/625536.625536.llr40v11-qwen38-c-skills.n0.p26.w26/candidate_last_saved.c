#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

void tsvc_2_s3110_fp64(const double *restrict aa, double *restrict bb, const int64_t LEN_2D) {
    const int64_t N = LEN_2D;
    const int64_t total = N * N;

    /* Global winner: max value, tie -> smallest row-major linear index.
       Initialize to (-INFINITY, INT64_MAX) so it never beats a real element. */
    double gmax = -INFINITY;
    int64_t gidx = INT64_MAX;

    int nt = omp_get_max_threads();
    if (nt < 1) nt = 1;

    double *tmax = (double *)malloc(sizeof(double) * (size_t)nt);
    int64_t *tidx = (int64_t *)malloc(sizeof(int64_t) * (size_t)nt);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        double lmax = -INFINITY;
        int64_t lidx = INT64_MAX;

        #pragma omp for schedule(static)
        for (int64_t k = 0; k < total; ++k) {
            const double v = aa[k];
            if (v > lmax) { lmax = v; lidx = k; }
        }

        tmax[tid] = lmax;
        tidx[tid] = lidx;
    }

    for (int t = 0; t < nt; ++t) {
        if (tmax[t] > gmax) { gmax = tmax[t]; gidx = tidx[t]; }
        else if (tmax[t] == gmax && tidx[t] < gidx) { gidx = tidx[t]; }
    }

    const int64_t xindex = gidx / N;
    const int64_t yindex = gidx - xindex * N;
    const double chksum = gmax + (double)xindex + (double)yindex;
    bb[0] = chksum;

    free(tmax);
    free(tidx);
}
