#include <stdint.h>
#include <omp.h>

static void k1_serial(double* restrict a, const double* restrict b, const double* restrict c, int64_t n) {
    for (int64_t i = 1; i < n; ++i) {
        a[i] = 0.75 * a[i - 1] + b[i] * c[i];
    }
}

static void k1_parallel(double* restrict a, const double* restrict b, const double* restrict c, int64_t n) {
    int nt = omp_get_max_threads();
    if (nt > 256) nt = 256;
    int64_t chunk = (n + nt - 1) / nt;

    double local_last[256];
    double factor[256];
    double carry[256];

    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num();
        int64_t start = (int64_t)tid * chunk;
        int64_t end = start + chunk;
        if (end > n) end = n;
        int64_t i0 = (tid == 0) ? 1 : start;

        double prev = 0.0;
        for (int64_t i = i0; i < end; ++i) {
            prev = 0.75 * prev + b[i] * c[i];
            a[i] = prev;
        }
        local_last[tid] = prev;

        double f = 1.0;
        double base = 0.75;
        int64_t m = end - i0;
        while (m > 0) {
            if (m & 1) f *= base;
            base *= base;
            m >>= 1;
        }
        factor[tid] = f;
    }

    carry[0] = a[0];
    for (int t = 1; t < nt; ++t) {
        carry[t] = factor[t - 1] * carry[t - 1] + local_last[t - 1];
    }

    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num();
        int64_t start = (int64_t)tid * chunk;
        int64_t end = start + chunk;
        if (end > n) end = n;
        int64_t i0 = (tid == 0) ? 1 : start;
        double P = carry[tid];
        double f = 0.75;
        for (int64_t i = i0; i < end; ++i) {
            a[i] += f * P;
            f *= 0.75;
        }
    }
}

static void kgt1_parallel(double* restrict a, const double* restrict b, const double* restrict c, int64_t K, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < K; ++j) {
        for (int64_t i = j + K; i < n; i += K) {
            a[i] = 0.75 * a[i - K] + b[i] * c[i];
        }
    }
}

void versioned_distance_update_fp64(double* restrict a, double* restrict b, double* restrict c, int64_t K, int64_t LEN_1D, uint8_t* restrict workspace, int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;
    if (K <= 0 || K >= LEN_1D) return;
    if (K == 1) {
        if (LEN_1D < 16384) {
            k1_serial(a, b, c, LEN_1D);
        } else {
            k1_parallel(a, b, c, LEN_1D);
        }
    } else {
        kgt1_parallel(a, b, c, K, LEN_1D);
    }
}
