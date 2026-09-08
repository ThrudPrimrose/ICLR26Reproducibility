#include <stdint.h>
#include <float.h>
#include <omp.h>

void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index,
                            double *restrict out_value, const int64_t LEN_1D) {
    if (LEN_1D <= 0) {
        out_value[0] = 0.0;
        out_index[0] = 0;
        return;
    }
    if (LEN_1D == 1) {
        out_value[0] = a[0];
        out_index[0] = 0;
        return;
    }

    /* Small inputs: sequential scan avoids parallel overhead. */
    if (LEN_1D < 4096) {
        double x = a[0];
        int64_t idx = 0;
        for (int64_t i = 1; i < LEN_1D; ++i) {
            if (a[i] > x) {
                x = a[i];
                idx = i;
            }
        }
        out_value[0] = x;
        out_index[0] = idx;
        return;
    }

    int max_threads = omp_get_max_threads();
    int nthreads = max_threads;
    int64_t min_chunk = 1024;
    int64_t cap = (LEN_1D + min_chunk - 1) / min_chunk;
    if (cap < nthreads) nthreads = (int)cap;
    if (nthreads < 1) nthreads = 1;

    double vals[256];
    int64_t idxs[256];

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < nthreads; ++t) {
        int64_t start = (t * LEN_1D) / nthreads;
        int64_t end   = ((t + 1) * LEN_1D) / nthreads;
        double x = a[start];
        int64_t idx = start;
        for (int64_t i = start + 1; i < end; ++i) {
            if (a[i] > x) {
                x = a[i];
                idx = i;
            }
        }
        vals[t] = x;
        idxs[t] = idx;
    }

    double x = vals[0];
    int64_t idx = idxs[0];
    for (int t = 1; t < nthreads; ++t) {
        double y = vals[t];
        int64_t j = idxs[t];
        if (y > x || (y == x && j < idx)) {
            x = y;
            idx = j;
        }
    }

    out_value[0] = x;
    out_index[0] = idx;
}
