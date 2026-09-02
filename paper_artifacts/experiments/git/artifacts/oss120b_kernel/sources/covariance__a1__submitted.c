#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <string.h>
#include <stdio.h>
#include <cblas.h>
#include <dlfcn.h>

void covariance_fp64(double *restrict cov, double *restrict data, int64_t M, int64_t N, double float_n) {
    // Allocate mean array and compute column-wise sum sequentially to match reference order
    double *mean = (double *)malloc((size_t)M * sizeof(double));
    if (!mean) return;
    // Compute mean: sum each column over rows and divide by N
    for (int64_t col = 0; col < M; ++col) {
        double sum = 0.0;
        for (int64_t row = 0; row < N; ++row) {
            sum += data[row * M + col];
        }
        mean[col] = sum / (double)N;
    }
    // Center the data (subtract column means)
    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < N; ++row) {
        double *row_ptr = data + row * M;
        for (int64_t col = 0; col < M; ++col) {
            row_ptr[col] -= mean[col];
        }
    }
    // Compute covariance matrix using BLAS dsyrk via dynamic loading for portability.
    // Try to load the OpenBLAS library at runtime.
    void *handle = dlopen("libopenblas.so", RTLD_LAZY);
    if (!handle) {
        // Fallback: simple serial implementation.
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = i; j < M; ++j) {
                double sum = 0.0;
                for (int64_t k = 0; k < N; ++k) {
                    sum += data[k * M + i] * data[k * M + j];
                }
                double val = sum / (float_n - 1.0);
                cov[i * M + j] = val;
                cov[j * M + i] = val;
            }
        }
    } else {
        // Set OpenBLAS threading to match available OpenMP threads for best performance.
        int max_threads = omp_get_max_threads();
        char buf[16];
        snprintf(buf, sizeof(buf), "%d", max_threads);
        setenv("OPENBLAS_NUM_THREADS", buf, 1);
        // Resolve the dsyrk symbol.
        void (*dsyrk_ptr)(enum CBLAS_ORDER, enum CBLAS_UPLO, enum CBLAS_TRANSPOSE,
                          int, int, double, const double*, int, double,
                          double*, int);
        *(void **)(&dsyrk_ptr) = dlsym(handle, "cblas_dsyrk");
        if (dsyrk_ptr) {
            dsyrk_ptr(CblasRowMajor, CblasUpper, CblasTrans,
                      (int)M, (int)N,
                      1.0/(float_n - 1.0), // alpha
                      data, (int)M,        // A and leading dimension (M columns)
                      0.0,                 // beta
                      cov, (int)M);        // C and leading dimension (M columns)
            // Copy upper triangular to lower triangular (symmetrize)
                        #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < M; ++i) {
                for (int64_t j = i; j < M; ++j) {
                    double val = cov[i * M + j];
                    cov[i * M + j] = val;
                    cov[j * M + i] = val;
                }
            }
        } else {
            // Fallback if symbol not found.
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < M; ++i) {
                for (int64_t j = i; j < M; ++j) {
                    double sum = 0.0;
                    for (int64_t k = 0; k < N; ++k) {
                        sum += data[k * M + i] * data[k * M + j];
                    }
                    double val = sum / (float_n - 1.0);
                    cov[i * M + j] = val;
                    cov[j * M + i] = val;
                }
            }
        }
        dlclose(handle);
    }
}
