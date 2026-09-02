#include <stdint.h>
#include <omp.h>

void wf_triangular_fp64(double *restrict a, const int64_t LEN_2D) {
    const int64_t n = LEN_2D;
    if (n < 2) return;

    if (n < 128) {
        for (int64_t i = 1; i < n; ++i) {
            for (int64_t j = i; j < n; ++j) {
                a[i * n + j] = a[i * n + j] + a[(i - 1) * n + j] + a[i * n + (j - 1)];
            }
        }
        return;
    }

    const int64_t B = 60;
    const int64_t nb = (n + B - 1) / B;

    #pragma omp parallel
    {
        for (int64_t s = 0; s <= 2 * (nb - 1); ++s) {
            const int64_t bi_min = (s <= nb - 1) ? 0 : s - (nb - 1);
            const int64_t bi_max = (s / 2 < nb) ? s / 2 : nb - 1;
            if (bi_min > bi_max) continue;

            #pragma omp for schedule(static)
            for (int64_t bi = bi_min; bi <= bi_max; ++bi) {
                const int64_t bj = s - bi;
                if (bj < bi) continue;

                const int64_t base = bi * B;
                const int64_t i0 = (base == 0) ? 1 : base;
                const int64_t i1 = (base + B < n) ? base + B : n;
                if (i0 >= i1) continue;

                if (bj > bi) {
                    const int64_t j0 = bj * B;
                    const int64_t j1 = (j0 + B < n) ? j0 + B : n;
                    if (j0 >= j1) continue;
                    for (int64_t i = i0; i < i1; ++i) {
                        double *restrict row = a + i * n;
                        const double *restrict nrow = a + (i - 1) * n;
                        for (int64_t j = j0; j < j1; ++j) {
                            row[j] = row[j] + nrow[j] + row[j - 1];
                        }
                    }
                } else { // diagonal block: bj == bi
                    const int64_t j1 = i1;
                    for (int64_t i = i0; i < j1; ++i) {
                        double *restrict row = a + i * n;
                        const double *restrict nrow = a + (i - 1) * n;
                        for (int64_t j = i; j < j1; ++j) {
                            row[j] = row[j] + nrow[j] + row[j - 1];
                        }
                    }
                }
            }
        }
    }
}
