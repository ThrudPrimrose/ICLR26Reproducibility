/* Host execution forced version: target region with if(0) and no explicit map clauses.
 * This registers a device kernel but runs the computation on the host, avoiding data transfers.
 */
#include <stdint.h>

void tsvc_2_vtvtv_fp64(double *restrict a, const double *restrict b,
                        const double *restrict c, const int64_t LEN_1D) {
    #pragma omp target if(0)
    {
        #pragma omp parallel for simd
        for (int64_t i = 0; i < LEN_1D; ++i) {
            a[i] = a[i] * b[i] * c[i];
        }
    }
}
