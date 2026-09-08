#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

#ifndef CHUNK_SIZE
#define CHUNK_SIZE 4096
#endif

static void serial_scan(const double *restrict a, double *restrict b, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        s += a[i];
        b[i] = s;
    }
}

void tsvc_2_s3112_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;

    int nt = omp_get_max_threads();
    if (nt <= 1 || LEN_1D < CHUNK_SIZE * 2) {
        serial_scan(a, b, LEN_1D);
        return;
    }

    const int64_t chunk = CHUNK_SIZE;
    int64_t nchunks = (LEN_1D + chunk - 1) / chunk;

    long double *restrict offsets = (long double *)malloc((size_t)nchunks * sizeof(long double));
    if (offsets == NULL) {
        serial_scan(a, b, LEN_1D);
        return;
    }

    /* Stage 1: per-chunk total sums in extended precision. */
    #pragma omp parallel for schedule(static)
    for (int64_t c = 0; c < nchunks; ++c) {
        int64_t start = c * chunk;
        int64_t end = start + chunk;
        if (end > LEN_1D) end = LEN_1D;
        long double s = 0.0L;
        for (int64_t i = start; i < end; ++i) {
            s += (long double)a[i];
        }
        offsets[c] = s;
    }

    /* Stage 2: exclusive prefix sum over chunk totals in extended precision. */
    long double s = 0.0L;
    for (int64_t c = 0; c < nchunks; ++c) {
        long double t = offsets[c];
        offsets[c] = s;
        s += t;
    }

    /* Stage 3: scan each chunk in extended precision, convert to double at store. */
    #pragma omp parallel for schedule(static)
    for (int64_t c = 0; c < nchunks; ++c) {
        int64_t start = c * chunk;
        int64_t end = start + chunk;
        if (end > LEN_1D) end = LEN_1D;
        long double s = offsets[c];
        for (int64_t i = start; i < end; ++i) {
            s += (long double)a[i];
            b[i] = (double)s;
        }
    }

    free(offsets);
}
