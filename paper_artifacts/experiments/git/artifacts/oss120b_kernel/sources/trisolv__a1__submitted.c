#include <stddef.h>
#include <stdint.h>
#include <cblas.h>

// Solve L * x = b where L is a dense lower-triangular matrix stored row-major.
// b: right-hand side, x: output vector (size N)
void trisolv_fp64(const double *restrict L, const double *restrict b, double *restrict x, int64_t N) {
    // Copy RHS b into output vector x (as BLAS overwrites input)
    for (int64_t i = 0; i < N; ++i) {
        x[i] = b[i];
    }
    // Use BLAS triangular solve (row-major, lower triangular, no transpose, non-unit diagonal)
    cblas_dtrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                (int)N, L, (int)N, x, 1);
}
