/* wf_triangular -- triangular north+west wavefront, row-serial exact-order v1.
 * a[i,j] = (a[i,j] + a[i-1,j]) + a[i,j-1]  for 1<=i<N, i<=j<N.
 * Exact same FP association as the reference (prev value kept in a register).
 */
#include <stdint.h>

void wf_triangular_fp64(double *restrict a, const int64_t LEN_2D,
                        unsigned char *restrict workspace,
                        const int64_t workspace_size)
{
    (void)workspace;
    (void)workspace_size;
    for (int64_t i = 1; i < LEN_2D; ++i) {
        double *restrict row = a + i * LEN_2D;
        const double *restrict up = a + (i - 1) * LEN_2D;
        double p = row[i - 1]; /* a[i, i-1] is never updated */
        for (int64_t j = i; j < LEN_2D; ++j) {
            double v = row[j] + up[j] + p; /* ((row[j] + up[j]) + p) == reference order */
            row[j] = v;
            p = v;
        }
    }
}
