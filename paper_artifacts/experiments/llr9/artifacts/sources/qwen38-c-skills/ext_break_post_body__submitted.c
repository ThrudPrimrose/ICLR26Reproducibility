#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <omp.h>

void ext_break_post_body_fp64(double *restrict a, const double *restrict b, const double *restrict c, int64_t LEN_1D) {
    if (LEN_1D <= 0) return;

    if (LEN_1D < 16384) {
        for (int64_t i = 0; i < LEN_1D; ++i) {
            a[i] = a[i] + b[i] * c[i];
            if (c[i] > b[i]) break;
        }
        return;
    }

    int64_t cut = INT64_MAX;

    #pragma omp parallel
    {
        #pragma omp for schedule(static) reduction(min:cut)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            int64_t v = (c[i] > b[i]) ? i : INT64_MAX;
            if (v < cut) cut = v;
        }

        int64_t n_update = (cut < LEN_1D) ? (cut + 1) : LEN_1D;

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n_update; ++i) {
            a[i] = a[i] + b[i] * c[i];
        }
    }
}
