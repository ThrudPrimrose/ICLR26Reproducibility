#include <stdint.h>
#include <math.h>
#include <omp.h>

void ludcmp_fp64(double *restrict A, const double *restrict b,
                 double *restrict x, double *restrict y, int64_t N) {
    if (N <= 0) return;

    /* LU factorization without pivoting (Doolittle, right-looking).
     * k is sequential; rows of the trailing update are independent. */
    for (int64_t k = 0; k < N; ++k) {
        const int64_t m = N - k - 1;
        if (m <= 0) break;
        const double inv = 1.0 / A[k * N + k];

        #pragma omp parallel
        {
            const int64_t nt = omp_get_num_threads();
            const int64_t tid = omp_get_thread_num();
            const int64_t per = (m + nt - 1) / nt;
            int64_t ist = k + 1 + tid * per;
            int64_t ien = ist + per;
            if (ien > N) ien = N;
            for (int64_t i = ist; i < ien; ++i) {
                double *col = A + i * N + k;
                double l = *col * inv;
                *col = l;
                const double *rk = (const double *) (A + k * N + k + 1);
                double *ri = (double *) (A + i * N + k + 1);
                for (int64_t j = 0; j < m; ++j)
                    ri[j] -= l * rk[j];
            }
        }
    }

    /* forward substitution: y = L^-1 b */
    for (int64_t i = 0; i < N; ++i) {
        double s = 0.0;
        const double *row = A + i * N;
        const double *yy = y;
        for (int64_t j = 0; j < i; ++j)
            s += row[j] * yy[j];
        y[i] = b[i] - s;
    }
    /* back substitution: x = U^-1 y */
    for (int64_t i = N - 1; i >= 0; --i) {
        double s = 0.0;
        const double *row = A + i * N;
        const double *xx = x;
        for (int64_t j = i + 1; j < N; ++j)
            s += row[j] * xx[j];
        x[i] = (y[i] - s) / row[i];
    }
}
