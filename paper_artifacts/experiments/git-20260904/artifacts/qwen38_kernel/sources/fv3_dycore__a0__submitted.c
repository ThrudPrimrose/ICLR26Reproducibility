#include <stdio.h>
#include <stdint.h>

void fv3_dycore_hord5_gt3_fp64(double *q, double *crx, double *cry,
                               double *x_area_flux, double *y_area_flux,
                               double *q_x_flux, double *q_y_flux,
                               double *dxa, double *dya, double *area,
                               int64_t s1, int64_t s2, int64_t s3,
                               int64_t s4, int64_t s5,
                               uint8_t *ws, int64_t ws_bytes) {
    (void)ws;
    long long nhalo=s1, ni=s3, nj=s4, nk=s5;
    long long ny=nj+2*nhalo, nz=nk;
    long long SI=ny*nz, SJ=nz;
    printf("INTS nhalo=%lld x=%lld ni=%lld nj=%lld nk=%lld => nx=%lld ny=%lld nz=%lld\n",
        nhalo,s2,ni,nj,nk, ni+2*nhalo, ny, nz);
    double *A[10] = {q,crx,cry,x_area_flux,y_area_flux,q_x_flux,q_y_flux,dxa,dya,area};
    long long offs[8] = {0,1,nz,2*nz,SI,SI+1,SI+nz,2*SI};
    for (int a=0;a<10;a++){
        printf("A%d:",a);
        for (int t=0;t<8;t++) printf(" %.9g", A[a][offs[t]]);
        printf("\n");
    }
    fflush(stdout);
}
