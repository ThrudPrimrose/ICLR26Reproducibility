/* TSVC tsvc_2 kernel s1232 (C)
 *
 * numpy reference (grader):
 *   for j in range(LEN_2D):
 *       for i in range(j * VLEN, LEN_2D):
 *           aa[i, j] = bb[i, j] + cc[i, j]
 *
 * C-track layout assumption: C-order flat buffer, aa[i,j] <-> buf[i*N + j].
 * Swapped view: row i updates columns j = 0 .. min(N-1, i/VLEN)  (VLEN <= 0 -> all columns)
 */
#include <stdint.h>
#include <omp.h>

void tsvc_2_s1232_fp64(double *aa, double *bb, double *cc,
                       int64_t LEN_2D, int64_t VLEN)
{
    const int64_t N = LEN_2D;
    if (N <= 0) return;

    const double *const restrict b = bb;
    const double *const restrict c = cc;

    #pragma omp parallel for schedule(dynamic, 128)
    for (int64_t i = 0; i < N; i++) {
        int64_t w;
        if (VLEN <= 0) {
            w = N;
        } else {
            w = i / VLEN + 1;
            if (w > N) w = N;
        }
        double *const restrict ar = aa + i * N;
        const double *const restrict br = b + i * N;
        const double *const restrict cr = c + i * N;
        for (int64_t j = 0; j < w; j++)
            ar[j] = br[j] + cr[j];
    }
}
