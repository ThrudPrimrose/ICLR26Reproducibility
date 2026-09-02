#include <stdint.h>
#include <omp.h>

void tsvc_2_s3112_fp64(double *restrict a, double *restrict b, int64_t n,
                       uint8_t *restrict workspace, int64_t workspace_bytes) {
    if (n <= 0) return;

    int nt = omp_get_max_threads();
    if (n < 4096 || nt <= 1 || workspace_bytes < nt * (int64_t)sizeof(double)) {
        double sum = 0.0;
        for (int64_t i = 0; i < n; i++) {
            sum += a[i];
            b[i] = sum;
        }
        return;
    }

    double *restrict offsets = (double *)workspace;

    int64_t block = (n + nt - 1) / nt;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int64_t i0 = tid * block;
        int64_t i1 = i0 + block;
        if (i1 > n) i1 = n;

        double sum = 0.0;
        for (int64_t i = i0; i < i1; i++) {
            sum += a[i];
        }
        offsets[tid] = sum;
    }

    double acc = 0.0;
    for (int t = 0; t < nt; t++) {
        double s = offsets[t];
        offsets[t] = acc;
        acc += s;
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int64_t i0 = tid * block;
        int64_t i1 = i0 + block;
        if (i1 > n) i1 = n;

        double sum = offsets[tid];
        for (int64_t i = i0; i < i1; i++) {
            sum += a[i];
            b[i] = sum;
        }
    }
}
