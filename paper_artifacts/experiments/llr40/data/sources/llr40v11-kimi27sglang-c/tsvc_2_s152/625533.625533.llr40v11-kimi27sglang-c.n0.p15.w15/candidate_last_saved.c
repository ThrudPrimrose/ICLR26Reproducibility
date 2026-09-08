#include <stdint.h>
#include <omp.h>

void tsvc_2_s152_fp64(double *restrict a, double *restrict b, const double *restrict c,
                      const double *restrict d, const double *restrict e, const int64_t LEN_1D) {
    const int64_t n = LEN_1D & ~15LL;
#pragma omp parallel for schedule(static) if(LEN_1D > 4096)
    for (int64_t i = 0; i < n; i += 16) {
        const double b0  = d[i+0]  * e[i+0];
        const double b1  = d[i+1]  * e[i+1];
        const double b2  = d[i+2]  * e[i+2];
        const double b3  = d[i+3]  * e[i+3];
        const double b4  = d[i+4]  * e[i+4];
        const double b5  = d[i+5]  * e[i+5];
        const double b6  = d[i+6]  * e[i+6];
        const double b7  = d[i+7]  * e[i+7];
        const double b8  = d[i+8]  * e[i+8];
        const double b9  = d[i+9]  * e[i+9];
        const double b10 = d[i+10] * e[i+10];
        const double b11 = d[i+11] * e[i+11];
        const double b12 = d[i+12] * e[i+12];
        const double b13 = d[i+13] * e[i+13];
        const double b14 = d[i+14] * e[i+14];
        const double b15 = d[i+15] * e[i+15];
        b[i+0]  = b0;  a[i+0]  += b0  * c[i+0];
        b[i+1]  = b1;  a[i+1]  += b1  * c[i+1];
        b[i+2]  = b2;  a[i+2]  += b2  * c[i+2];
        b[i+3]  = b3;  a[i+3]  += b3  * c[i+3];
        b[i+4]  = b4;  a[i+4]  += b4  * c[i+4];
        b[i+5]  = b5;  a[i+5]  += b5  * c[i+5];
        b[i+6]  = b6;  a[i+6]  += b6  * c[i+6];
        b[i+7]  = b7;  a[i+7]  += b7  * c[i+7];
        b[i+8]  = b8;  a[i+8]  += b8  * c[i+8];
        b[i+9]  = b9;  a[i+9]  += b9  * c[i+9];
        b[i+10] = b10; a[i+10] += b10 * c[i+10];
        b[i+11] = b11; a[i+11] += b11 * c[i+11];
        b[i+12] = b12; a[i+12] += b12 * c[i+12];
        b[i+13] = b13; a[i+13] += b13 * c[i+13];
        b[i+14] = b14; a[i+14] += b14 * c[i+14];
        b[i+15] = b15; a[i+15] += b15 * c[i+15];
    }
    for (int64_t i = n; i < LEN_1D; ++i) {
        const double bi = d[i] * e[i];
        b[i] = bi;
        a[i] += bi * c[i];
    }
}
