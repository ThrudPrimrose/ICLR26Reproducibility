#include <stdint.h>
#include <omp.h>

static int64_t lower_bound(const int64_t *row_ptr, int64_t n, int64_t target)
{
    int64_t lo = 0;
    int64_t hi = n;
    while (lo < hi) {
        const int64_t mid = (lo + hi) >> 1;
        if (row_ptr[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

void segment_reduce_ragged_fp64(double *restrict out,
                                int64_t *restrict row_ptr,
                                double *restrict val,
                                double *restrict w,
                                int64_t NSEG,
                                uint8_t *restrict workspace,
                                int64_t workspace_size)
{
    (void)workspace;
    (void)workspace_size;

    const int64_t total = row_ptr[NSEG];
    const int nt = omp_get_max_threads();

    int64_t starts[1024];
    starts[0] = 0;
    if (nt > 1) {
        for (int t = 1; t < nt; ++t) {
            starts[t] = lower_bound(row_ptr, NSEG + 1, (t * total) / nt);
        }
    }
    starts[nt] = NSEG;

    #pragma omp parallel num_threads(nt) shared(row_ptr, val, w, out, NSEG, total, starts)
    {
        const int tid = omp_get_thread_num();
        const int64_t s_begin = starts[tid];
        const int64_t s_end = starts[tid + 1];

        for (int64_t s = s_begin; s < s_end; ++s) {
            double acc = 0.0;
            const int64_t e0 = row_ptr[s];
            const int64_t e1 = row_ptr[s + 1];
            for (int64_t e = e0; e < e1; ++e) {
                acc += val[e] * w[e];
            }
            out[s] = acc;
        }
    }
}
