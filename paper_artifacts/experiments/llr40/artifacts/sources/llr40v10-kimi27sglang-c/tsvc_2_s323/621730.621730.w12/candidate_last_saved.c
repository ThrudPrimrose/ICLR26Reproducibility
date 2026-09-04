#include <stdint.h>
#include <omp.h>

void tsvc_2_s323_fp64(double *restrict a, double *restrict b, const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
    if (LEN_1D <= 1) return;

    const int64_t n = LEN_1D - 1;

    /* For small problems, the sequential recurrence is fastest. */
    if (n < 4096) {
        double b_prev = b[0];
        for (int64_t i = 1; i < LEN_1D; ++i) {
            const double ci = c[i];
            const double ai = b_prev + ci * d[i];
            a[i] = ai;
            b_prev = ai + ci * e[i];
            b[i] = b_prev;
        }
        return;
    }

    const int max_threads = omp_get_max_threads();
    double locals[max_threads];

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        const int64_t chunk = (n + nt - 1) / nt;
        const int64_t start = 1 + (int64_t)tid * chunk;
        int64_t end = start + chunk;
        if (end > LEN_1D) end = LEN_1D;

        /* Step 1: compute s[i] = c[i]*d[i] + c[i]*e[i] into b[i]. */
        #pragma omp simd
        for (int64_t i = start; i < end; ++i) {
            const double ci = c[i];
            b[i] = ci * d[i] + ci * e[i];
        }

        /* Step 2a: local inclusive prefix sum within the chunk. */
        double sum = 0.0;
        if (start < end) {
            sum = b[start];
            for (int64_t i = start + 1; i < end; ++i) {
                sum += b[i];
                b[i] = sum;
            }
        }
        locals[tid] = sum;

        #pragma omp barrier

        /* Step 2b: prefix sum of chunk totals (single thread, protected). */
        #pragma omp single
        {
            double offset = b[0];
            for (int t = 0; t < nt; ++t) {
                const double local_sum = locals[t];
                locals[t] = offset;
                offset += local_sum;
            }
        }

        #pragma omp barrier

        /* Step 2c: add chunk offset and compute a[i] = b[i-1] + c[i]*d[i]. */
        const double off = locals[tid];
        if (start < end) {
            for (int64_t i = start; i < end; ++i) {
                b[i] += off;
            }
            #pragma omp simd
            for (int64_t i = start; i < end; ++i) {
                a[i] = b[i - 1] + c[i] * d[i];
            }
        }
    }
}
