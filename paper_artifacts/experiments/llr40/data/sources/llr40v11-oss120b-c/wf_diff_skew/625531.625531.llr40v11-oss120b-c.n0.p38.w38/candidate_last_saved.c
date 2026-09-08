/* Optimized wf_diff_skew kernel using OpenMP tasks with dependencies.
   Computes a[i][j] = a[i][j] + a[i-1][j] + a[i-1][j+1]
   for i = 1..LEN_2D-1, j = 0..LEN_2D-2.
   The inner loop is SIMD- vectorized. Tasks enforce row-level dependencies.
*/
#include <stdint.h>

void wf_diff_skew_fp64(double *restrict a, const int64_t LEN_2D) {
    const int64_t n = LEN_2D - 1; // inner loop length
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int64_t i = 1; i < LEN_2D; ++i) {
                // depend on the whole previous row and produce the whole current row
                #pragma omp task depend(in: a[(i-1)*LEN_2D:LEN_2D]) depend(out: a[i*LEN_2D:LEN_2D])
                {
                    double *restrict ai = a + i * LEN_2D;
                    double *restrict ap = a + (i - 1) * LEN_2D;
                    #pragma omp simd
                    for (int64_t j = 0; j < n; ++j) {
                        ai[j] = ai[j] + ap[j] + ap[j + 1];
                    }
                }
            }
        }
    }
}
