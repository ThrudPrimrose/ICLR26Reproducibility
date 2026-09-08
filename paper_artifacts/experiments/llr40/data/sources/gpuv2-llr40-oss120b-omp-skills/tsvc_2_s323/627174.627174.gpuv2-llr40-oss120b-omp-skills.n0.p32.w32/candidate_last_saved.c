#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void tsvc_2_s323_fp64(double *restrict a, double *restrict b,
                       const double *restrict c, const double *restrict d,
                       const double *restrict e, const int64_t LEN_1D) {
#if defined(__clang__) && defined(_OPENMP)
    // GPU version using a parallel prefix sum (scan) without OpenMP 5.0 inscan.
    // Allocate temporary array for per‑iteration increments.
    double *inc = (double *)malloc((size_t)LEN_1D * sizeof(double));
    if (!inc) {
        // Fallback to serial on allocation failure.
        for (int64_t i = 1; i < LEN_1D; ++i) {
            a[i] = b[i - 1] + c[i] * d[i];
            b[i] = a[i] + c[i] * e[i];
        }
        return;
    }
    const int MAX_THREADS = 1024;
    double *block_sum = (double *)malloc(MAX_THREADS * sizeof(double));
    if (!block_sum) {
        free(inc);
        for (int64_t i = 1; i < LEN_1D; ++i) {
            a[i] = b[i - 1] + c[i] * d[i];
            b[i] = a[i] + c[i] * e[i];
        }
        return;
    }

    #pragma omp target map(to: c[0:LEN_1D], d[0:LEN_1D], e[0:LEN_1D]) \
                     map(tofrom: a[0:LEN_1D], b[0:LEN_1D]) \
                     map(tofrom: inc[0:LEN_1D], block_sum[0:MAX_THREADS])
    {
        // 1) Compute increments: inc[i] = c[i] * (d[i] + e[i]), i>=1; inc[0]=0.
        #pragma omp parallel for
        for (int64_t i = 1; i < LEN_1D; ++i) {
            inc[i] = c[i] * (d[i] + e[i]);
        }
        inc[0] = 0.0;

        double b0 = b[0];

        // 2) Parallel prefix sum of inc using per‑thread exclusive offsets.
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            int64_t start = (LEN_1D * tid) / nt;
            int64_t end   = (LEN_1D * (tid + 1)) / nt;
            // Local prefix within the thread's chunk.
            double cur = 0.0;
            for (int64_t i = start; i < end; ++i) {
                cur += inc[i];
                inc[i] = cur; // store local prefix (without previous chunks)
            }
            block_sum[tid] = cur; // total sum of this chunk
            #pragma omp barrier
            // Compute exclusive offsets for each chunk.
            #pragma omp single
            {
                double offset = 0.0;
                for (int i = 0; i < nt; ++i) {
                    double sum_i = block_sum[i];
                    block_sum[i] = offset;   // replace with exclusive offset
                    offset += sum_i;
                }
            }
            #pragma omp barrier
            double add = block_sum[tid];
            // Apply offset and produce final a and b.
            for (int64_t i = start; i < end; ++i) {
                double inc_val = inc[i] + add;
                double bi = b0 + inc_val;
                b[i] = bi;
                a[i] = bi - c[i] * e[i];
            }
        }
    }

    free(inc);
    free(block_sum);

    // Verify that execution occurred on the device; otherwise fall back to serial.
    int on_device = 0;
    #pragma omp target map(from: on_device)
    on_device = !omp_is_initial_device();
    if (!on_device) {
        for (int64_t i = 1; i < LEN_1D; ++i) {
            a[i] = b[i - 1] + c[i] * d[i];
            b[i] = a[i] + c[i] * e[i];
        }
    }
#else
    // Serial fallback.
    for (int64_t i = 1; i < LEN_1D; ++i) {
        a[i] = b[i - 1] + c[i] * d[i];
        b[i] = a[i] + c[i] * e[i];
    }
#endif
}
