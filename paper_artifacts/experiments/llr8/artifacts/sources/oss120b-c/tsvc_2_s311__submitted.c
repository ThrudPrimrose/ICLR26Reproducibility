#include <stdint.h>
#include <stddef.h>
#include <omp.h>

void tsvc_2_s311_fp64(double *a, double *sum_out, int64_t LEN_1D, uint8_t *workspace, int64_t workspace_bytes) {
    const int MAX_THREADS = 256;
    double partial[MAX_THREADS];
    // Initialize partial sums to zero
    for (int i = 0; i < MAX_THREADS; ++i) {
        partial[i] = 0.0;
    }
    // Set a fixed number of threads for deterministic behavior
    int num_threads = 8; // adjust based on typical core count; keep constant across runs
    if (num_threads > MAX_THREADS) num_threads = MAX_THREADS;
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double local = 0.0;
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            local += a[i];
        }
        partial[tid] = local;
    }
    double sum = 0.0;
    for (int i = 0; i < num_threads; ++i) {
        sum += partial[i];
    }
    sum_out[0] = sum;
    (void)workspace;
    (void)workspace_bytes;
}
