#include <stdint.h>
#include <omp.h>

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb,
                       const double *restrict cc, const int64_t LEN_2D) {
    const int64_t n = LEN_2D;
    if (n <= 8) return;

#pragma omp parallel
    {
        const int64_t nt = omp_get_num_threads();
        const int64_t tid = omp_get_thread_num();
        const int64_t m = n - 8;
        const int64_t base = m / nt;
        const int64_t rem = m % nt;
        const int64_t start = 8 + tid * base + (tid < rem ? tid : rem);
        const int64_t end = start + base + (tid < rem ? 1 : 0);

        for (int64_t r = 8; r < n; ++r) {
            const int64_t row = r * n;
            const int64_t prev = (r - 1) * n;
#pragma omp simd
            for (int64_t k = start; k < end; ++k) {
                const double c = cc[row + k];
                aa[row + k] = aa[prev + k] + c;
                bb[row + k] = bb[prev + k] + c;
            }
        }
    }
}
