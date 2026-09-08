#include <stddef.h>
#include <stdint.h>
#include <omp.h>

void compact_threshold_pack_fp64(int64_t * restrict out_count,
                                  double * restrict packed,
                                  double * restrict src,
                                  double * restrict weight,
                                  int64_t LEN_1D,
                                  uint8_t * restrict workspace,
                                  int64_t workspace_bytes) {
    int max_threads = omp_get_max_threads();
    int64_t capacity = workspace_bytes / sizeof(int64_t);

    if (workspace == NULL || capacity < (int64_t)max_threads) {
        int64_t n = 0;
        for (int64_t i = 0; i < LEN_1D; i++) {
            double s = src[i];
            if (s > 0.0) {
                packed[n] = s * weight[i];
                n++;
            }
        }
        out_count[0] = n;
        return;
    }

    int nt = max_threads;
    if ((int64_t)nt > capacity) nt = (int)capacity;
    int64_t * restrict counts = (int64_t * restrict)__builtin_assume_aligned((void *)workspace, 256);

    #pragma omp parallel num_threads(nt) default(none) \
            shared(counts, src, weight, packed, out_count, LEN_1D, nt)
    {
        int tid = omp_get_thread_num();
        int64_t chunk = (LEN_1D + nt - 1) / nt;
        int64_t start = (int64_t)tid * chunk;
        int64_t end = start + chunk;
        if (end > LEN_1D) end = LEN_1D;

        int64_t c = 0;
        for (int64_t i = start; i < end; i++) {
            if (src[i] > 0.0) c++;
        }
        counts[tid] = c;

        #pragma omp barrier

        #pragma omp single
        {
            int64_t acc = 0;
            for (int i = 0; i < nt; i++) {
                int64_t tmp = counts[i];
                counts[i] = acc;
                acc += tmp;
            }
            out_count[0] = acc;
        }

        int64_t offset = counts[tid];
        for (int64_t i = start; i < end; i++) {
            double s = src[i];
            if (s > 0.0) {
                packed[offset] = s * weight[i];
                offset++;
            }
        }
    }
}
