#include <stdint.h>
#include <omp.h>

static inline void run_xpos_gt10(double *restrict a, double *restrict b, double *restrict c,
                                 const double *restrict d, const double *restrict e,
                                 const int64_t LEN_1D) {
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (a[i] > b[i]) {
            a[i] += b[i] * d[i];
            c[i] += d[i] * d[i];
        } else {
            b[i] = a[i] + e[i] * e[i];
            c[i] = a[i] + d[i] * d[i];
        }
    }
}

static inline void run_xpos_le10(double *restrict a, double *restrict b, double *restrict c,
                                 const double *restrict d, const double *restrict e,
                                 const int64_t LEN_1D) {
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (a[i] > b[i]) {
            a[i] += b[i] * d[i];
            c[i] = d[i] * e[i] + 1.0;
        } else {
            b[i] = a[i] + e[i] * e[i];
            c[i] = a[i] + d[i] * d[i];
        }
    }
}

static inline void run_xneg_gt10(double *restrict a, double *restrict b, double *restrict c,
                                 const double *restrict d, const double *restrict e,
                                 const int64_t LEN_1D) {
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (a[i] > b[i]) {
            a[i] += b[i] * d[i];
            c[i] += d[i] * d[i];
        } else {
            b[i] = a[i] + e[i] * e[i];
            c[i] += e[i] * e[i];
        }
    }
}

static inline void run_xneg_le10(double *restrict a, double *restrict b, double *restrict c,
                                 const double *restrict d, const double *restrict e,
                                 const int64_t LEN_1D) {
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (a[i] > b[i]) {
            a[i] += b[i] * d[i];
            c[i] = d[i] * e[i] + 1.0;
        } else {
            b[i] = a[i] + e[i] * e[i];
            c[i] += e[i] * e[i];
        }
    }
}

void tsvc_2_s2710_fp64(double *restrict a, double *restrict b, double *restrict c,
                       const double *restrict d, const double *restrict e,
                       const double *restrict x, const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;

    const int x_pos = x[0] > 0.0;

    if (x_pos) {
        if (LEN_1D > 10) {
            run_xpos_gt10(a, b, c, d, e, LEN_1D);
        } else {
            run_xpos_le10(a, b, c, d, e, LEN_1D);
        }
    } else {
        if (LEN_1D > 10) {
            run_xneg_gt10(a, b, c, d, e, LEN_1D);
        } else {
            run_xneg_le10(a, b, c, d, e, LEN_1D);
        }
    }
}
