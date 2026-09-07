#include <stdio.h>

/* ABI probe: print all args */
void fv3_dycore_fp64(double *a1, double *a2, double *a3, double *a4, double *a5,
                     double *a6, double *a7, double *a8, double *a9, double *a10,
                     double *a11, double *a12, double *a13,
                     long long n1, long long n2, long long n3, long long n4, long long n5) {
    double *p[13] = {a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13};
    for (int i = 0; i < 13; i++) {
        printf("a%d=%p v0=%g v1=%g v7=%g\n", i+1, p[i], p[i][0], p[i][1], p[i][7]);
    }
    printf("n1=%lld n2=%lld n3=%lld n4=%lld n5=%lld\n", n1, n2, n3, n4, n5);
    fflush(stdout);
}
