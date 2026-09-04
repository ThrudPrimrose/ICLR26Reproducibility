#include <stdint.h>
#include <omp.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

void wf_triangular_fp64(double *restrict a, const int64_t LEN_2D) {
    const int64_t n = LEN_2D;
    if (n <= 1) return;

    const int64_t B = BLOCK_SIZE;

    if (n < 2 * B) {
        for (int64_t d = 2; d <= 2 * n - 2; ++d) {
            int64_t i_min = (d > n - 1) ? d - (n - 1) : 1;
            int64_t i_max = d / 2;
            #pragma omp parallel for schedule(static)
            for (int64_t i = i_min; i <= i_max; ++i) {
                int64_t j = d - i;
                a[i * n + j] += a[(i - 1) * n + j] + a[i * n + (j - 1)];
            }
        }
        return;
    }

    const int64_t nt = (n + B - 1) / B;

    for (int64_t td = 0; td <= 2 * (nt - 1); ++td) {
        int64_t ti_min = (td > nt - 1) ? td - (nt - 1) : 0;
        int64_t ti_max = (td < nt - 1) ? td : nt - 1;
        #pragma omp parallel for schedule(static)
        for (int64_t ti = ti_min; ti <= ti_max; ++ti) {
            int64_t tj = td - ti;
            if (ti > tj) continue;

            int64_t i_start = ti * B;
            if (i_start < 1) i_start = 1;
            int64_t i_end = (ti + 1) * B - 1;
            if (i_end > n - 1) i_end = n - 1;

            int64_t j_start_base = tj * B;
            int64_t j_end_base = (tj + 1) * B - 1;
            if (j_end_base > n - 1) j_end_base = n - 1;

            for (int64_t i = i_start; i <= i_end; ++i) {
                int64_t j_start = j_start_base;
                if (j_start < i) j_start = i;
                int64_t j_end = j_end_base;
                if (j_end < j_start) continue;
                for (int64_t j = j_start; j <= j_end; ++j) {
                    a[i * n + j] += a[(i - 1) * n + j] + a[i * n + (j - 1)];
                }
            }
        }
    }
}
