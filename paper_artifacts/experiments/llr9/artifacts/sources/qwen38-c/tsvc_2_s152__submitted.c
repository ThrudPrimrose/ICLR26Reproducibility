#include <stdint.h>

void tsvc_2_s152_fp64(double* a, double* b, double* c, double* d, double* e,
                      int64_t LEN_1D, uint8_t* ws, int64_t ws_bytes)
{
    (void)ws; (void)ws_bytes;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D; i++) {
        double t = d[i] * e[i];
        b[i] = t;
        a[i] = a[i] + t * c[i];
    }
}
