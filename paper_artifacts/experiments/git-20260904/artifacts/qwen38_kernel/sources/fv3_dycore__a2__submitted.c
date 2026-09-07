#include <stdio.h>
#include <stdint.h>
void fv3_dycore_hord5_gt3_fp64(double *area, double *crx, double *cry,
                               double *dxa, double *dya, double *q,
                               double *q_x_flux, double *q_y_flux,
                               double *x_area_flux, double *y_area_flux,
                               int64_t grid_type, int64_t hord, int64_t ni,
                               int64_t nj, int64_t nk,
                               uint8_t *ws, int64_t ws_bytes) {
    printf("grid_type=%lld hord=%lld ni=%lld nj=%lld nk=%lld ws=%p ws_bytes=%lld\n",
           (long long)grid_type,(long long)hord,(long long)ni,(long long)nj,(long long)nk,
           (void*)ws,(long long)ws_bytes);
    /* confirm array order via values */
    double *p[10]={area,crx,cry,dxa,dya,q,q_x_flux,q_y_flux,x_area_flux,y_area_flux};
    for(int i=0;i<10;i++) printf("arg%d=%g\n", i+1, p[i][0]);
    fflush(stdout);
}
