#include <stdint.h>
#include <omp.h>

static void __attribute__((noinline))
tsvc_2_s152_par(double *restrict ar, double *restrict br,
                const double *restrict cr, const double *restrict dr,
                const double *restrict er, int64_t len_1d)
{
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < len_1d; i++) br[i] = dr[i] * er[i];
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < len_1d; i++) ar[i] = ar[i] + br[i] * cr[i];
}

void tsvc_2_s152_fp64(double *a, double *b, double *c, double *d, double *e,
                      int64_t len_1d, uint8_t *workspace, int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;

    double *restrict ar = a;
    double *restrict br = b;
    const double *restrict cr = (const double *restrict)c;
    const double *restrict dr = (const double *restrict)d;
    const double *restrict er = (const double *restrict)e;

    if (len_1d >= 32768 && omp_get_max_threads() > 1) {
        tsvc_2_s152_par(ar, br, cr, dr, er, len_1d);
        return;
    }
    for (int64_t i = 0; i < len_1d; i++) br[i] = dr[i] * er[i];
    for (int64_t i = 0; i < len_1d; i++) ar[i] = ar[i] + br[i] * cr[i];
}
