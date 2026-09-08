#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void ext_war_unit_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
    const int64_t N = LEN_1D - 1;
    if (N <= 0) return;

    if (N < 200000) {
        for (int64_t i = 0; i < N; ++i)
            a[i] = a[i + 1] + b[i];
        return;
    }

    const int64_t SUPER = 4194304;
    size_t tmp_bytes = (size_t)SUPER * sizeof(double);
    double *restrict tmp = (double *)aligned_alloc(64, tmp_bytes);
    if (!tmp) return;

    #pragma omp parallel
    {
        for (int64_t s = 0; s < N; s += SUPER) {
            int64_t e = (s + SUPER < N) ? s + SUPER : N;

            #pragma omp for schedule(static)
            for (int64_t i = s; i < e; ++i)
                tmp[i - s] = a[i + 1];

            #pragma omp for schedule(static)
            for (int64_t i = s; i < e; ++i)
                a[i] = tmp[i - s] + b[i];
        }
    }

    free(tmp);
}
