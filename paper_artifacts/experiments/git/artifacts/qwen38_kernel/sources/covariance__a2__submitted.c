/* Covariance kernel: cov = (data - colmean)^T (data - colmean) / (float_n - 1)
 * data: N x M row-major double, modified in place (centered)
 * cov:  M x M row-major double
 * v1: OpenBLAS dsyrk via dlopen (row-major, upper triangle, Trans).
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>
#include <omp.h>

typedef void (*dsyrk_fn)(int, int, int, int64_t, int64_t, double, const double *, int64_t, double, double *, int64_t);
typedef int  (*setn_fn)(int);

static dsyrk_fn g_dsyrk = 0;
static int g_blas_ready = 0;

static void init_blas(void) {
    if (g_blas_ready) return;
    g_blas_ready = 1;
    void *h = dlopen("libopenblas.so.0", RTLD_NOW);
    if (!h) h = dlopen("libopenblas.so", RTLD_NOW);
    if (!h) h = dlopen("libblas.so.3", RTLD_NOW);
    if (!h) return;
    g_dsyrk = (dsyrk_fn)dlsym(h, "cblas_dsyrk");
    if (!g_dsyrk) return;
    setn_fn setn = (setn_fn)dlsym(h, "openblas_set_num_threads");
    if (setn) {
        int nt = omp_get_max_threads();
        if (nt < 1) nt = 1;
        setn(nt);
    }
}

/* Column means: mean[m] = sum_l data[l*M+m] / N.  Tiled over 8 columns, parallel over column tiles. */
static void colmeans(const double *restrict data, int64_t M, int64_t N, double *restrict mean) {
    const double invn = 1.0 / (double)N;
    #pragma omp parallel for schedule(static)
    for (int64_t m0 = 0; m0 < M; m0 += 8) {
        const int64_t m1 = m0 + 8 < M ? m0 + 8 : M;
        double s[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int64_t l = 0; l < N; l++) {
            const double *r = data + l * M + m0;
            for (int64_t m = m0; m < m1; m++) s[m - m0] += r[m - m0];
        }
        for (int64_t m = m0; m < m1; m++) mean[m] = s[m - m0] * invn;
    }
}

/* Center in place: data[l*M+m] -= mean[m]. */
static void center(double *restrict data, int64_t M, int64_t N, const double *restrict mean) {
    #pragma omp parallel for schedule(static)
    for (int64_t l = 0; l < N; l++) {
        double *r = data + l * M;
        for (int64_t m = 0; m < M; m++) r[m] -= mean[m];
    }
}

/* Mirror upper triangle to lower. */
static void mirror(double *restrict cov, int64_t M) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 1; i < M; i++) {
        for (int64_t j = 0; j < i; j++) cov[i * M + j] = cov[j * M + i];
    }
}

void covariance_fp64(double *restrict cov, double *restrict data, int64_t M, int64_t N, double float_n) {
    if (M <= 0 || N <= 0) return;
    init_blas();
    double *mean = (double *)malloc((size_t)M * sizeof(double));
    colmeans(data, M, N, mean);
    center(data, M, N, mean);
    free(mean);

    const double scale = 1.0 / (float_n - 1.0);

    if (g_dsyrk) {
        /* RowMajor, Upper(121), Trans(112): C(n x n) = A^T A, A stored k x n row-major (k=N, n=M, lda=M). */
        g_dsyrk(101, 121, 112, (int64_t)M, (int64_t)N, scale, data, (int64_t)M, 0.0, cov, (int64_t)M);
    } else {
        /* Fallback: plain scalar triple loop (upper triangle). */
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < M; i++) {
            for (int64_t j = i; j < M; j++) {
                double s = 0.0;
                for (int64_t l = 0; l < N; l++) s += data[l * M + i] * data[l * M + j];
                cov[i * M + j] = s * scale;
            }
        }
    }
    mirror(cov, M);
}
