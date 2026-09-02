#include <stdint.h>

/* TSVC s252:
 *   t = 0; for i: s = b[i]*c[i]; a[i] = s + t; t = s
 * i.e. a[0] = b[0]*c[0];  a[i] = b[i]*c[i] + b[i-1]*c[i-1]  (i >= 1)
 * The scalar t is a 1-step rotation, so the loop is fully parallel.
 */
void tsvc_2_s252_fp64(double *restrict a, const double *restrict b,
                      const double *restrict c, int64_t n)
{
    if (n <= 0) return;
    a[0] = b[0] * c[0] + 0.0;
    if (n <= 65536) {
        for (int64_t i = 1; i < n; i++) {
            a[i] = b[i] * c[i] + b[i - 1] * c[i - 1];
        }
        return;
    }
    #pragma omp parallel for schedule(static)
    for (int64_t i = 1; i < n; i++) {
        a[i] = b[i] * c[i] + b[i - 1] * c[i - 1];
    }
}
