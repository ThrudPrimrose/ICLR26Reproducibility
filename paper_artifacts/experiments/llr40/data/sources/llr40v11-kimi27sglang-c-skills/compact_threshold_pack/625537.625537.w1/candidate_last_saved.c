#include <stdint.h>
#include <omp.h>

void compact_threshold_pack_fp64(int64_t *restrict out_count,
                                   const double *restrict src,
                                   const double *restrict weight,
                                   double *restrict packed,
                                   int64_t LEN_1D,
                                   uint8_t *restrict workspace,
                                   int64_t workspace_size)
{
    (void)workspace_size;

    if (LEN_1D <= 0) {
        out_count[0] = 0;
        return;
    }

    int max_threads = omp_get_max_threads();
    int64_t *restrict counts = (int64_t *)workspace;
    int64_t *restrict offsets = counts + max_threads;

    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int64_t start = (int64_t)tid * LEN_1D / nthreads;
        int64_t end = (int64_t)(tid + 1) * LEN_1D / nthreads;

        int64_t c = 0;
        for (int64_t i = start; i < end; i++) {
            if (src[i] > 0.0) c++;
        }
        counts[tid] = c;

        #pragma omp barrier
        #pragma omp single
        {
            int64_t total = 0;
            for (int t = 0; t < nthreads; t++) {
                offsets[t] = total;
                total += counts[t];
            }
            out_count[0] = total;
        }
        #pragma omp barrier

        int64_t pos = offsets[tid];
        for (int64_t i = start; i < end; i++) {
            double s = src[i];
            if (s > 0.0) {
                packed[pos++] = s * weight[i];
            }
        }
    }
}
