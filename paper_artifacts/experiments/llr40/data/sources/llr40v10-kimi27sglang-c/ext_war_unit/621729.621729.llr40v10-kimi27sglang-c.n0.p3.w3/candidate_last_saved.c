#include <stdint.h>
#include <stdlib.h>

static double *boundary = NULL;
static int64_t boundary_cap = 0;

void ext_war_unit_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
    const int64_t n = LEN_1D - 1;
    if (n <= 0) return;

    if (n < 4096) {
        #pragma omp simd
        for (int64_t i = 0; i < n; ++i) {
            a[i] = a[i + 1] + b[i];
        }
        return;
    }

    const int64_t CHUNK = 65536;
    const int64_t nch = (n + CHUNK - 1) / CHUNK;

    if (nch > boundary_cap) {
        double *p = (double *)realloc(boundary, (size_t)nch * sizeof(double));
        if (p) {
            boundary = p;
            boundary_cap = nch;
        }
    }
    if (!boundary) {
        #pragma omp simd
        for (int64_t i = 0; i < n; ++i) {
            a[i] = a[i + 1] + b[i];
        }
        return;
    }

    for (int64_t c = 0; c < nch; ++c) {
        const int64_t end = (c + 1) * CHUNK;
        boundary[c] = a[end <= n ? end : n];
    }

    #pragma omp parallel for schedule(static)
    for (int64_t c = 0; c < nch; ++c) {
        const int64_t start = c * CHUNK;
        const int64_t last = start + CHUNK;
        const int64_t end = last <= n ? last : n;

        #pragma omp simd
        for (int64_t i = start; i < end - 1; ++i) {
            a[i] = a[i + 1] + b[i];
        }
        a[end - 1] = boundary[c] + b[end - 1];
    }
}
