#include <stdint.h>
#include <omp.h>

static inline int64_t lower_bound(const int64_t *arr, int64_t n, int64_t key)
{
    int64_t lo = 0, hi = n;
    while (lo < hi) {
        int64_t mid = (lo + hi) >> 1;
        if (arr[mid] < key)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

void segment_reduce_ragged_fp64(double *restrict out,
                                int64_t *restrict row_ptr,
                                double *restrict val,
                                double *restrict w,
                                int64_t NSEG,
                                uint8_t *restrict workspace,
                                int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;

    const int64_t total = row_ptr[NSEG];

    #pragma omp parallel
    {
        const int nt = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int64_t t_start = (int64_t)tid * total / nt;
        const int64_t t_end   = (int64_t)(tid + 1) * total / nt;
        const int64_t s_begin = lower_bound(row_ptr, NSEG + 1, t_start);
        const int64_t s_end   = lower_bound(row_ptr, NSEG + 1, t_end);

        for (int64_t s = s_begin; s < s_end; ++s) {
            int64_t e0 = row_ptr[s];
            int64_t e1 = row_ptr[s + 1];
            double acc = 0.0;
            #pragma omp simd reduction(+:acc)
            for (int64_t e = e0; e < e1; ++e) {
                acc += val[e] * w[e];
            }
            out[s] = acc;
        }
    }
}
