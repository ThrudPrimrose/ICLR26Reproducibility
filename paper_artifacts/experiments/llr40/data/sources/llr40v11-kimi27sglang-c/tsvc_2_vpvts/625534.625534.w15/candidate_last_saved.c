#include <stdint.h>
#include <limits.h>
#include <cblas.h>

void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b,
                       const int64_t LEN_1D, const int64_t S)
{
    const double alpha = (double)S;
    int64_t rem = LEN_1D;
    int64_t off = 0;

    while (rem > 0) {
        int chunk = (rem > INT_MAX) ? INT_MAX : (int)rem;
        cblas_daxpy(chunk, alpha, b + off, 1, a + off, 1);
        off += chunk;
        rem -= chunk;
    }
}
