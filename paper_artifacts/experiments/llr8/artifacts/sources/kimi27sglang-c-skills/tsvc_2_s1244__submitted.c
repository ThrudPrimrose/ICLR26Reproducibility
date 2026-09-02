#include <stdint.h>
#include <omp.h>

void tsvc_2_s1244_fp64(double *restrict a, double *restrict b,
                       double *restrict c, double *restrict d,
                       int64_t n, uint8_t *restrict workspace,
                       int64_t workspace_bytes) {
    if (n <= 1) return;
    int64_t need = 8 * n;
    if (workspace && workspace_bytes >= need) {
        double *restrict olda = (double *restrict)workspace;
        #pragma omp parallel for simd schedule(static)
        for (int64_t i = 0; i < n; i++) {
            olda[i] = a[i];
        }
        #pragma omp parallel for simd schedule(static)
        for (int64_t i = 0; i < n - 1; i++) {
            double bi = b[i];
            double ci = c[i];
            double ai = bi + ci * ci + bi * bi + ci;
            a[i] = ai;
            d[i] = ai + olda[i + 1];
        }
    } else {
        for (int64_t i = 0; i < n - 1; i++) {
            double bi = b[i];
            double ci = c[i];
            double ai = bi + ci * ci + bi * bi + ci;
            a[i] = ai;
            d[i] = ai + a[i + 1];
        }
    }
}
