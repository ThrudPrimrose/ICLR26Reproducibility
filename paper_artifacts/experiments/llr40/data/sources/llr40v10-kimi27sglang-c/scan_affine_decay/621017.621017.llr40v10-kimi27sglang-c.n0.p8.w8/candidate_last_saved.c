#include <stdint.h>
#include <omp.h>
#include <stdlib.h>

void scan_affine_decay_fp64(double *restrict c, double *restrict x, double *restrict y,
                            int64_t LEN_1D, uint8_t *restrict workspace, int64_t workspace_size) {
    int64_t n = LEN_1D;
    if (n <= 0) return;
    if (n <= 1024) {
        y[0] = x[0];
        for (int64_t i = 1; i < n; ++i) {
            y[i] = c[i] * y[i - 1] + x[i];
        }
        return;
    }
    int64_t bsize = 4096;
    int64_t nb = (n + bsize - 1) / bsize;
    size_t need_bytes = (size_t)nb * 3 * sizeof(double);

    double *buf;
    if (workspace != NULL && (size_t)workspace_size >= need_bytes) {
        buf = (double*)(void*)workspace;
    } else {
        static __thread double *tls_buf = NULL;
        static __thread size_t tls_cap = 0;
        if (tls_cap < need_bytes) {
            double *p = (double*)realloc(tls_buf, need_bytes);
            if (p) {
                tls_buf = p;
                tls_cap = need_bytes;
            }
        }
        buf = tls_buf;
    }
    if (buf == NULL) {
        y[0] = x[0];
        for (int64_t i = 1; i < n; ++i) {
            y[i] = c[i] * y[i - 1] + x[i];
        }
        return;
    }

    double *P = buf;
    double *S = buf + nb;
    double *carry = buf + 2*nb;

    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < nb; ++b) {
        int64_t l = b * bsize;
        int64_t r = l + bsize;
        if (r > n) r = n;
        double prod = 1.0;
        double sum = 0.0;
        for (int64_t i = l; i < r; ++i) {
            prod *= c[i];
            sum = sum * c[i] + x[i];
        }
        P[b] = prod;
        S[b] = sum;
    }

    carry[0] = 0.0;
    for (int64_t b = 1; b < nb; ++b) {
        carry[b] = S[b-1] + P[b-1] * carry[b-1];
    }

    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < nb; ++b) {
        int64_t l = b * bsize;
        int64_t r = l + bsize;
        if (r > n) r = n;
        double prev = carry[b];
        for (int64_t i = l; i < r; ++i) {
            y[i] = c[i] * prev + x[i];
            prev = y[i];
        }
    }
}
