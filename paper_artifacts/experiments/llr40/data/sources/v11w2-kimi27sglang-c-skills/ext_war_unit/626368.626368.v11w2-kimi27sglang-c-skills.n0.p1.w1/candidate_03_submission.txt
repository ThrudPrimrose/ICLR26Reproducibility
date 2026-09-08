#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void ext_war_unit_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
    if (LEN_1D <= 1) return;
    
    size_t bytes = (size_t)LEN_1D * sizeof(double);
    double *restrict tmp = (double *)malloc(bytes);
    if (tmp == NULL) return;
    
    #pragma omp parallel
    {
        #pragma omp for simd schedule(static)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            tmp[i] = a[i];
        }
        
        #pragma omp for simd schedule(static)
        for (int64_t i = 0; i < LEN_1D - 1; ++i) {
            a[i] = tmp[i + 1] + b[i];
        }
    }
    
    free(tmp);
}
