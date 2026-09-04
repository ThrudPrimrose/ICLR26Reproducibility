#include <stdint.h>
#include <math.h>
#include <omp.h>

void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index,
                            double *restrict out_value, const int64_t LEN_1D) {
    const int64_t n = LEN_1D;
    if (n <= 0) { *out_value = 0.0; *out_index = 0; return; }

    int maxt = omp_get_max_threads();
    int nt = maxt;
    /* keep at least ~1024 elements per thread */
    int64_t want = (n + 1023) / 1024;
    if (nt > want) nt = (int)want;
    if (nt < 1) nt = 1;
    if (nt > 1024) nt = 1024;

    static double local_max[1024];
    static int64_t local_idx[1024];

    #pragma omp parallel num_threads(nt)
    {
        const int tid = omp_get_thread_num();
        const int64_t chunk = (n + nt - 1) / nt;
        int64_t start = (int64_t)tid * chunk;
        int64_t end = start + chunk;
        if (end > n) end = n;
        if (start >= n) {
            local_max[tid] = -INFINITY;
            local_idx[tid] = -1;
        } else {
            double m = a[start];
            int64_t mi = start;
            for (int64_t i = start + 1; i < end; ++i) {
                double v = a[i];
                if (v > m) { m = v; mi = i; }
            }
            local_max[tid] = m;
            local_idx[tid] = mi;
        }
    }

    double best = 0.0;
    int64_t besti = 0;
    int first = 1;
    for (int t = 0; t < nt; ++t) {
        if (local_idx[t] < 0) continue;
        if (first || local_max[t] > best ||
            (local_max[t] == best && local_idx[t] < besti)) {
            best = local_max[t]; besti = local_idx[t]; first = 0;
        }
    }
    *out_value = best;
    *out_index = besti;
}
