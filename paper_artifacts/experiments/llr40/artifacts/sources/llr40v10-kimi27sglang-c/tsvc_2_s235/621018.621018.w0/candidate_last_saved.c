#include <stdint.h>
#include <omp.h>

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa,
                      const double *restrict b, const double *restrict bb,
                      const double *restrict c, const int64_t LEN_2D) {
    const int64_t n = LEN_2D;

    #pragma omp parallel default(none) shared(a, aa, b, bb, c, n)
    {
        const int nt = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int64_t chunk = (n + nt - 1) / nt;
        const int64_t i0 = (int64_t)tid * chunk;
        const int64_t i1 = (i0 + chunk < n) ? (i0 + chunk) : n;

        if (i0 < n) {
            #pragma omp simd
            for (int64_t i = i0; i < i1; ++i) {
                a[i] += b[i] * c[i];
            }

            for (int64_t j = 1; j < n; ++j) {
                #pragma omp simd
                for (int64_t i = i0; i < i1; ++i) {
                    aa[j * n + i] = aa[(j - 1) * n + i] + bb[j * n + i] * a[i];
                }
            }
        }
    }
}
