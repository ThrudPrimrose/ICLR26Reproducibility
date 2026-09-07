#include <stdint.h>
void finite_volume_transport(double *restrict q,
                             double *restrict crx,
                             double *restrict cry,
                             double *restrict x_area_flux,
                             double *restrict y_area_flux,
                             double *restrict q_x_flux,
                             double *restrict q_y_flux,
                             double *restrict dxa,
                             double *restrict dya,
                             double *restrict area,
                             const int64_t nhalo,
                             const int64_t ni,
                             const int64_t nj,
                             const int64_t nk,
                             const int64_t hord,
                             const int64_t grid_type) {
    (void)q;(void)crx;(void)cry;(void)x_area_flux;(void)y_area_flux;
    (void)q_x_flux;(void)q_y_flux;(void)dxa;(void)dya;(void)area;
    (void)nhalo;(void)ni;(void)nj;(void)nk;(void)hord;(void)grid_type;
}
