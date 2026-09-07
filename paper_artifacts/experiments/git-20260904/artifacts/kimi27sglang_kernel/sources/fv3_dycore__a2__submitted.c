#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#ifndef NH
#define NH 3
#endif

#define IDX(i,j,k) (((int64_t)(i))*ny + (j))*nk + (k)

static const double P1 =  0.5833333333333334;
static const double P2 = -0.08333333333333333;
static const double C1 = -0.14285714285714285;
static const double C2 =  0.7857142857142857;
static const double C3 =  0.35714285714285715;

static inline double *align_ptr(uint8_t *p) {
    uintptr_t a = (uintptr_t)p;
    a = (a + 63) & ~(uintptr_t)63;
    return (double *)a;
}

static void copy_corners_x(double * restrict f,
                           int64_t nx, int64_t ny, int64_t nk)
{
    #pragma omp parallel for
    for (int64_t k = 0; k < nk; ++k) {
        f[IDX(0,0,k)] = f[IDX(0,5,k)];
        f[IDX(0,1,k)] = f[IDX(1,5,k)];
        f[IDX(0,2,k)] = f[IDX(2,5,k)];
        f[IDX(1,0,k)] = f[IDX(0,4,k)];
        f[IDX(1,1,k)] = f[IDX(1,4,k)];
        f[IDX(1,2,k)] = f[IDX(2,4,k)];
        f[IDX(2,0,k)] = f[IDX(0,3,k)];
        f[IDX(2,1,k)] = f[IDX(1,3,k)];
        f[IDX(2,2,k)] = f[IDX(2,3,k)];

        f[IDX(0,ny-4,k)] = f[IDX(2,ny-7,k)];
        f[IDX(0,ny-3,k)] = f[IDX(1,ny-7,k)];
        f[IDX(0,ny-2,k)] = f[IDX(0,ny-7,k)];
        f[IDX(1,ny-4,k)] = f[IDX(2,ny-6,k)];
        f[IDX(1,ny-3,k)] = f[IDX(1,ny-6,k)];
        f[IDX(1,ny-2,k)] = f[IDX(0,ny-6,k)];
        f[IDX(2,ny-4,k)] = f[IDX(2,ny-5,k)];
        f[IDX(2,ny-3,k)] = f[IDX(1,ny-5,k)];
        f[IDX(2,ny-2,k)] = f[IDX(0,ny-5,k)];

        f[IDX(nx-4,0,k)] = f[IDX(nx-2,3,k)];
        f[IDX(nx-4,1,k)] = f[IDX(nx-3,3,k)];
        f[IDX(nx-4,2,k)] = f[IDX(nx-4,3,k)];
        f[IDX(nx-3,0,k)] = f[IDX(nx-2,4,k)];
        f[IDX(nx-3,1,k)] = f[IDX(nx-3,4,k)];
        f[IDX(nx-3,2,k)] = f[IDX(nx-4,4,k)];
        f[IDX(nx-2,0,k)] = f[IDX(nx-2,5,k)];
        f[IDX(nx-2,1,k)] = f[IDX(nx-3,5,k)];
        f[IDX(nx-2,2,k)] = f[IDX(nx-4,5,k)];

        f[IDX(nx-4,ny-2,k)] = f[IDX(nx-2,ny-5,k)];
        f[IDX(nx-4,ny-3,k)] = f[IDX(nx-3,ny-5,k)];
        f[IDX(nx-4,ny-4,k)] = f[IDX(nx-4,ny-5,k)];
        f[IDX(nx-3,ny-2,k)] = f[IDX(nx-2,ny-6,k)];
        f[IDX(nx-3,ny-3,k)] = f[IDX(nx-3,ny-6,k)];
        f[IDX(nx-3,ny-4,k)] = f[IDX(nx-4,ny-6,k)];
        f[IDX(nx-2,ny-2,k)] = f[IDX(nx-2,ny-7,k)];
        f[IDX(nx-2,ny-3,k)] = f[IDX(nx-3,ny-7,k)];
        f[IDX(nx-2,ny-4,k)] = f[IDX(nx-4,ny-7,k)];
    }
}

static void copy_corners_y(double * restrict f,
                           int64_t nx, int64_t ny, int64_t nk)
{
    #pragma omp parallel for
    for (int64_t k = 0; k < nk; ++k) {
        f[IDX(0,0,k)] = f[IDX(5,0,k)];
        f[IDX(1,0,k)] = f[IDX(5,1,k)];
        f[IDX(2,0,k)] = f[IDX(5,2,k)];
        f[IDX(0,1,k)] = f[IDX(4,0,k)];
        f[IDX(1,1,k)] = f[IDX(4,1,k)];
        f[IDX(2,1,k)] = f[IDX(4,2,k)];
        f[IDX(0,2,k)] = f[IDX(3,0,k)];
        f[IDX(1,2,k)] = f[IDX(3,1,k)];
        f[IDX(2,2,k)] = f[IDX(3,2,k)];

        f[IDX(nx-4,0,k)] = f[IDX(nx-7,2,k)];
        f[IDX(nx-3,0,k)] = f[IDX(nx-7,1,k)];
        f[IDX(nx-2,0,k)] = f[IDX(nx-7,0,k)];
        f[IDX(nx-4,1,k)] = f[IDX(nx-6,2,k)];
        f[IDX(nx-3,1,k)] = f[IDX(nx-6,1,k)];
        f[IDX(nx-2,1,k)] = f[IDX(nx-6,0,k)];
        f[IDX(nx-4,2,k)] = f[IDX(nx-5,2,k)];
        f[IDX(nx-3,2,k)] = f[IDX(nx-5,1,k)];
        f[IDX(nx-2,2,k)] = f[IDX(nx-5,0,k)];

        f[IDX(0,ny-2,k)] = f[IDX(5,ny-2,k)];
        f[IDX(0,ny-3,k)] = f[IDX(4,ny-2,k)];
        f[IDX(0,ny-4,k)] = f[IDX(3,ny-2,k)];
        f[IDX(1,ny-2,k)] = f[IDX(5,ny-3,k)];
        f[IDX(1,ny-3,k)] = f[IDX(4,ny-3,k)];
        f[IDX(1,ny-4,k)] = f[IDX(3,ny-3,k)];
        f[IDX(2,ny-2,k)] = f[IDX(5,ny-4,k)];
        f[IDX(2,ny-3,k)] = f[IDX(4,ny-4,k)];
        f[IDX(2,ny-4,k)] = f[IDX(3,ny-4,k)];

        f[IDX(nx-2,ny-4,k)] = f[IDX(nx-5,ny-2,k)];
        f[IDX(nx-2,ny-3,k)] = f[IDX(nx-6,ny-2,k)];
        f[IDX(nx-2,ny-2,k)] = f[IDX(nx-7,ny-2,k)];
        f[IDX(nx-3,ny-4,k)] = f[IDX(nx-5,ny-3,k)];
        f[IDX(nx-3,ny-3,k)] = f[IDX(nx-6,ny-3,k)];
        f[IDX(nx-3,ny-2,k)] = f[IDX(nx-7,ny-3,k)];
        f[IDX(nx-4,ny-4,k)] = f[IDX(nx-5,ny-4,k)];
        f[IDX(nx-4,ny-3,k)] = f[IDX(nx-6,ny-4,k)];
        f[IDX(nx-4,ny-2,k)] = f[IDX(nx-7,ny-4,k)];
    }
}

static void compute_al_x(const double * restrict q, const double * restrict dxa,
                         double * restrict al,
                         int64_t nhalo, int64_t ni, int64_t nk,
                         int64_t nx, int64_t ny, int64_t grid_type)
{
    int64_t i_start = nhalo;
    int64_t i_end   = nhalo + ni - 1;
    int64_t lo = i_start - 1;
    int64_t hi = i_end + 3;

    #pragma omp parallel for collapse(2)
    for (int64_t i = lo; i < hi; ++i) {
        for (int64_t j = 0; j < ny; ++j) {
            const double * restrict qrowm2 = q + IDX(i-2,j,0);
            const double * restrict qrowm1 = q + IDX(i-1,j,0);
            const double * restrict qrow   = q + IDX(i,j,0);
            const double * restrict qrowp1 = q + IDX(i+1,j,0);
            double * restrict alrow = al + IDX(i,j,0);
            for (int64_t k = 0; k < nk; ++k) {
                alrow[k] = P1 * (qrowm1[k] + qrow[k]) +
                           P2 * (qrowm2[k] + qrowp1[k]);
            }
        }
    }

    if (grid_type < 3) {
        #pragma omp parallel for collapse(2)
        for (int64_t j = 0; j < ny; ++j) {
            for (int64_t k = 0; k < nk; ++k) {
                int64_t ia[2] = { i_start - 1, i_end };
                for (int t = 0; t < 2; ++t) {
                    int64_t ic = ia[t];
                    al[IDX(ic,j,k)] = C1 * q[IDX(ic-2,j,k)] +
                                      C2 * q[IDX(ic-1,j,k)] +
                                      C3 * q[IDX(ic,j,k)];
                }
                int64_t ib[2] = { i_start, i_end + 1 };
                for (int t = 0; t < 2; ++t) {
                    int64_t ic = ib[t];
                    double left = ((2.0 * dxa[IDX(ic-1,j,k)] + dxa[IDX(ic-2,j,k)]) * q[IDX(ic-1,j,k)]
                                   - dxa[IDX(ic-1,j,k)] * q[IDX(ic-2,j,k)])
                                  / (dxa[IDX(ic-2,j,k)] + dxa[IDX(ic-1,j,k)]);
                    double right = ((2.0 * dxa[IDX(ic,j,k)] + dxa[IDX(ic+1,j,k)]) * q[IDX(ic,j,k)]
                                    - dxa[IDX(ic,j,k)] * q[IDX(ic+1,j,k)])
                                   / (dxa[IDX(ic,j,k)] + dxa[IDX(ic+1,j,k)]);
                    al[IDX(ic,j,k)] = 0.5 * (left + right);
                }
                int64_t icc[2] = { i_start + 1, i_end + 2 };
                for (int t = 0; t < 2; ++t) {
                    int64_t ic = icc[t];
                    al[IDX(ic,j,k)] = C3 * q[IDX(ic-1,j,k)] +
                                      C2 * q[IDX(ic,j,k)] +
                                      C1 * q[IDX(ic+1,j,k)];
                }
            }
        }
    }
}

static void compute_al_y(const double * restrict q, const double * restrict dya,
                         double * restrict al,
                         int64_t nhalo, int64_t nj, int64_t nk,
                         int64_t nx, int64_t ny, int64_t grid_type)
{
    int64_t j_start = nhalo;
    int64_t j_end   = nhalo + nj - 1;
    int64_t lo = j_start - 1;
    int64_t hi = j_end + 3;

    #pragma omp parallel for collapse(2)
    for (int64_t j = lo; j < hi; ++j) {
        for (int64_t i = 0; i < nx; ++i) {
            const double * restrict qrowm2 = q + IDX(i,j-2,0);
            const double * restrict qrowm1 = q + IDX(i,j-1,0);
            const double * restrict qrow   = q + IDX(i,j,0);
            const double * restrict qrowp1 = q + IDX(i,j+1,0);
            double * restrict alrow = al + IDX(i,j,0);
            for (int64_t k = 0; k < nk; ++k) {
                alrow[k] = P1 * (qrowm1[k] + qrow[k]) +
                           P2 * (qrowm2[k] + qrowp1[k]);
            }
        }
    }

    if (grid_type < 3) {
        #pragma omp parallel for collapse(2)
        for (int64_t i = 0; i < nx; ++i) {
            for (int64_t k = 0; k < nk; ++k) {
                int64_t ja[2] = { j_start - 1, j_end };
                for (int t = 0; t < 2; ++t) {
                    int64_t jc = ja[t];
                    al[IDX(i,jc,k)] = C1 * q[IDX(i,jc-2,k)] +
                                      C2 * q[IDX(i,jc-1,k)] +
                                      C3 * q[IDX(i,jc,k)];
                }
                int64_t jb[2] = { j_start, j_end + 1 };
                for (int t = 0; t < 2; ++t) {
                    int64_t jc = jb[t];
                    double left = ((2.0 * dya[IDX(i,jc-1,k)] + dya[IDX(i,jc-2,k)]) * q[IDX(i,jc-1,k)]
                                   - dya[IDX(i,jc-1,k)] * q[IDX(i,jc-2,k)])
                                  / (dya[IDX(i,jc-2,k)] + dya[IDX(i,jc-1,k)]);
                    double right = ((2.0 * dya[IDX(i,jc,k)] + dya[IDX(i,jc+1,k)]) * q[IDX(i,jc,k)]
                                    - dya[IDX(i,jc,k)] * q[IDX(i,jc+1,k)])
                                   / (dya[IDX(i,jc,k)] + dya[IDX(i,jc+1,k)]);
                    al[IDX(i,jc,k)] = 0.5 * (left + right);
                }
                int64_t jcc[2] = { j_start + 1, j_end + 2 };
                for (int t = 0; t < 2; ++t) {
                    int64_t jc = jcc[t];
                    al[IDX(i,jc,k)] = C3 * q[IDX(i,jc-1,k)] +
                                      C2 * q[IDX(i,jc,k)] +
                                      C1 * q[IDX(i,jc+1,k)];
                }
            }
        }
    }
}

static inline int smt5_cond(int mord, double bl, double br, double b0)
{
    if (mord == 5) {
        return (bl * br < 0.0);
    } else {
        return (3.0 * fabs(b0) < fabs(bl - br));
    }
}

static void xppm_flux(const double * restrict q, const double * restrict courant,
                      const double * restrict al, double * restrict xflux,
                      int64_t nhalo, int64_t ni, int64_t nk,
                      int64_t nx, int64_t ny, int64_t mord)
{
    int64_t i_start = nhalo;
    int64_t i_end   = nhalo + ni - 1;
    int64_t lo = i_start;
    int64_t hi = i_end + 2;

    #pragma omp parallel for collapse(2)
    for (int64_t i = lo; i < hi; ++i) {
        for (int64_t j = 0; j < ny; ++j) {
            const double * restrict qrow  = q + IDX(i,j,0);
            const double * restrict qrowm = q + IDX(i-1,j,0);
            const double * restrict alrow = al + IDX(i,j,0);
            const double * restrict alrowp = al + IDX(i+1,j,0);
            const double * restrict alrowm = al + IDX(i-1,j,0);
            const double * restrict crow = courant + IDX(i,j,0);
            double * restrict frow = xflux + IDX(i,j,0);
            for (int64_t k = 0; k < nk; ++k) {
                double qi  = qrow[k];
                double qim = qrowm[k];
                double bl  = alrow[k]   - qi;
                double br  = alrowp[k]  - qi;
                double b0  = bl + br;
                double blm = alrowm[k]  - qim;
                double brm = alrow[k]   - qim;
                double b0m = blm + brm;

                int smt5  = smt5_cond((int)mord, bl, br, b0);
                int smt5m = smt5_cond((int)mord, blm, brm, b0m);
                double mask = (smt5 || smt5m) ? 1.0 : 0.0;

                double c = crow[k];
                if (c > 0.0) {
                    frow[k] = qim + (1.0 - c) * (brm - c * b0m) * mask;
                } else {
                    frow[k] = qi  + (1.0 + c) * (bl  + c * b0)  * mask;
                }
            }
        }
    }
}

static void yppm_flux(const double * restrict q, const double * restrict courant,
                      const double * restrict al, double * restrict yflux,
                      int64_t nhalo, int64_t nj, int64_t nk,
                      int64_t nx, int64_t ny, int64_t mord)
{
    int64_t j_start = nhalo;
    int64_t j_end   = nhalo + nj - 1;
    int64_t lo = j_start;
    int64_t hi = j_end + 2;

    #pragma omp parallel for collapse(2)
    for (int64_t j = lo; j < hi; ++j) {
        for (int64_t i = 0; i < nx; ++i) {
            const double * restrict qrow  = q + IDX(i,j,0);
            const double * restrict qrowm = q + IDX(i,j-1,0);
            const double * restrict alrow = al + IDX(i,j,0);
            const double * restrict alrowp = al + IDX(i,j+1,0);
            const double * restrict alrowm = al + IDX(i,j-1,0);
            const double * restrict crow = courant + IDX(i,j,0);
            double * restrict frow = yflux + IDX(i,j,0);
            for (int64_t k = 0; k < nk; ++k) {
                double qj  = qrow[k];
                double qjm = qrowm[k];
                double bl  = alrow[k]   - qj;
                double br  = alrowp[k]  - qj;
                double b0  = bl + br;
                double blm = alrowm[k]  - qjm;
                double brm = alrow[k]   - qjm;
                double b0m = blm + brm;

                int smt5  = smt5_cond((int)mord, bl, br, b0);
                int smt5m = smt5_cond((int)mord, blm, brm, b0m);
                double mask = (smt5 || smt5m) ? 1.0 : 0.0;

                double c = crow[k];
                if (c > 0.0) {
                    frow[k] = qjm + (1.0 - c) * (brm - c * b0m) * mask;
                } else {
                    frow[k] = qj  + (1.0 + c) * (bl  + c * b0)  * mask;
                }
            }
        }
    }
}

static void q_i_stencil(const double * restrict q, const double * restrict area,
                        const double * restrict y_area_flux,
                        const double * restrict q_advected_along_y,
                        double * restrict q_i,
                        int64_t nk,
                        int64_t nx, int64_t ny)
{
    int64_t j0 = 3;
    int64_t j1 = ny - 3;
    #pragma omp parallel for collapse(2)
    for (int64_t j = j0; j < j1; ++j) {
        for (int64_t i = 0; i < nx; ++i) {
            const double * restrict qrow = q + IDX(i,j,0);
            const double * restrict arow = area + IDX(i,j,0);
            const double * restrict yfrow = y_area_flux + IDX(i,j,0);
            const double * restrict yfrowp = y_area_flux + IDX(i,j+1,0);
            const double * restrict qadv = q_advected_along_y + IDX(i,j,0);
            const double * restrict qadvp = q_advected_along_y + IDX(i,j+1,0);
            double * restrict out = q_i + IDX(i,j,0);
            for (int64_t k = 0; k < nk; ++k) {
                double fyy_j   = yfrow[k]   * qadv[k];
                double fyy_jp1 = yfrowp[k]  * qadvp[k];
                double denom = arow[k] + yfrow[k] - yfrowp[k];
                out[k] = (qrow[k] * arow[k] + fyy_j - fyy_jp1) / denom;
            }
        }
    }
}

static void q_j_stencil(const double * restrict q, const double * restrict area,
                        const double * restrict x_area_flux,
                        const double * restrict fx2,
                        double * restrict q_j,
                        int64_t nk,
                        int64_t nx, int64_t ny)
{
    int64_t i0 = 3;
    int64_t i1 = nx - 3;
    #pragma omp parallel for collapse(2)
    for (int64_t i = i0; i < i1; ++i) {
        for (int64_t j = 0; j < ny; ++j) {
            const double * restrict qrow = q + IDX(i,j,0);
            const double * restrict arow = area + IDX(i,j,0);
            const double * restrict xfrow = x_area_flux + IDX(i,j,0);
            const double * restrict xfrowp = x_area_flux + IDX(i+1,j,0);
            const double * restrict fx2row = fx2 + IDX(i,j,0);
            const double * restrict fx2rowp = fx2 + IDX(i+1,j,0);
            double * restrict out = q_j + IDX(i,j,0);
            for (int64_t k = 0; k < nk; ++k) {
                double fx1_i   = xfrow[k]  * fx2row[k];
                double fx1_ip1 = xfrowp[k] * fx2rowp[k];
                double denom = arow[k] + xfrow[k] - xfrowp[k];
                out[k] = (qrow[k] * arow[k] + fx1_i - fx1_ip1) / denom;
            }
        }
    }
}

static void final_fluxes(const double * restrict q_ayxa, const double * restrict q_xa,
                         const double * restrict q_axya, const double * restrict q_ya,
                         const double * restrict x_unit_flux, const double * restrict y_unit_flux,
                         double * restrict x_flux, double * restrict y_flux,
                         int64_t nhalo, int64_t ni, int64_t nj, int64_t nk,
                         int64_t nx, int64_t ny)
{
    int64_t i_start = nhalo, i_end = nhalo + ni - 1;
    int64_t j_start = nhalo, j_end = nhalo + nj - 1;

    #pragma omp parallel for collapse(2)
    for (int64_t i = i_start; i < i_end + 2; ++i) {
        for (int64_t j = j_start; j < j_end + 1; ++j) {
            int64_t base = IDX(i,j,0);
            const double * restrict ay = q_ayxa + base;
            const double * restrict xa = q_xa + base;
            const double * restrict xuf = x_unit_flux + base;
            double * restrict xf = x_flux + base;
            for (int64_t k = 0; k < nk; ++k) {
                xf[k] = 0.5 * (ay[k] + xa[k]) * xuf[k];
            }
        }
    }
    #pragma omp parallel for collapse(2)
    for (int64_t i = i_start; i < i_end + 1; ++i) {
        for (int64_t j = j_start; j < j_end + 2; ++j) {
            int64_t base = IDX(i,j,0);
            const double * restrict ax = q_axya + base;
            const double * restrict ya = q_ya + base;
            const double * restrict yuf = y_unit_flux + base;
            double * restrict yf = y_flux + base;
            for (int64_t k = 0; k < nk; ++k) {
                yf[k] = 0.5 * (ax[k] + ya[k]) * yuf[k];
            }
        }
    }
}

void fv3_dycore_hord5_gt3_fp64(double * restrict q,
                               double * restrict crx,
                               double * restrict cry,
                               double * restrict x_area_flux,
                               double * restrict y_area_flux,
                               double * restrict q_x_flux,
                               double * restrict q_y_flux,
                               double * restrict dxa,
                               double * restrict dya,
                               double * restrict area,
                               int64_t ni, int64_t nj, int64_t nk,
                               int64_t hord, int64_t grid_type,
                               uint8_t * restrict workspace,
                               int64_t workspace_bytes)
{
    (void)hord;
    const int64_t nhalo = NH;
    int64_t nx = nhalo + ni + nhalo;
    int64_t ny = nhalo + nj + nhalo;
    int64_t nxyz = nx * ny * nk;

    memset(q_x_flux, 0, (size_t)(nxyz * sizeof(double)));
    memset(q_y_flux, 0, (size_t)(nxyz * sizeof(double)));

    double *q_y_mean, *q_x_mean, *q_advected, *q_ayxa, *q_axya, *al;
    size_t need = 6 * ((size_t)nxyz + 8) * sizeof(double);
    if (workspace && workspace_bytes >= (int64_t)need) {
        q_y_mean  = align_ptr(workspace);
        q_x_mean  = q_y_mean + nxyz + 8;
        q_advected = q_x_mean + nxyz + 8;
        q_ayxa    = q_advected + nxyz + 8;
        q_axya    = q_ayxa + nxyz + 8;
        al        = q_axya + nxyz + 8;
    } else {
        q_y_mean = (double *)malloc(6 * (size_t)nxyz * sizeof(double));
        q_x_mean = q_y_mean + nxyz;
        q_advected = q_x_mean + nxyz;
        q_ayxa = q_advected + nxyz;
        q_axya = q_ayxa + nxyz;
        al = q_axya + nxyz;
    }

    int64_t ord_outer = hord;
    int64_t ord_inner = (hord == 10) ? 8 : hord;

    /* Path 1: y advection then x advection */
    copy_corners_y(q, nx, ny, nk);
    compute_al_y(q, dya, al, nhalo, nj, nk, nx, ny, grid_type);
    yppm_flux(q, cry, al, q_y_mean, nhalo, nj, nk, nx, ny, ord_inner);
    q_i_stencil(q, area, y_area_flux, q_y_mean, q_advected, nk, nx, ny);
    compute_al_x(q_advected, dxa, al, nhalo, ni, nk, nx, ny, grid_type);
    xppm_flux(q_advected, crx, al, q_ayxa, nhalo, ni, nk, nx, ny, ord_outer);

    /* Path 2: x advection then y advection */
    copy_corners_x(q, nx, ny, nk);
    compute_al_x(q, dxa, al, nhalo, ni, nk, nx, ny, grid_type);
    xppm_flux(q, crx, al, q_x_mean, nhalo, ni, nk, nx, ny, ord_inner);
    q_j_stencil(q, area, x_area_flux, q_x_mean, q_advected, nk, nx, ny);
    compute_al_y(q_advected, dya, al, nhalo, nj, nk, nx, ny, grid_type);
    yppm_flux(q_advected, cry, al, q_axya, nhalo, nj, nk, nx, ny, ord_outer);

    final_fluxes(q_ayxa, q_x_mean, q_axya, q_y_mean,
                 x_area_flux, y_area_flux, q_x_flux, q_y_flux,
                 nhalo, ni, nj, nk, nx, ny);

    if (!workspace || workspace_bytes < (int64_t)need) {
        free(q_y_mean);
    }
}
