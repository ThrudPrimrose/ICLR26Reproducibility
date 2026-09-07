#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifndef NH
#define NH 3
#endif

/* Layout: index = ((i)*ny + (j))*nk + (k)  (C-order, k contiguous) */
#define IDX(i,j,k) (((int64_t)(i))*ny + (j))*nk + (k)

static const double P1 =  0.5833333333333334;
static const double P2 = -0.08333333333333333;
static const double C1 = -0.14285714285714285;
static const double C2 =  0.7857142857142857;
static const double C3 =  0.35714285714285715;

/* Corner copies for q (3x3 corner blocks), applied for every k level. */
static void copy_corners_x(double * restrict f,
                           int64_t nx, int64_t ny, int64_t nk)
{
    for (int64_t k = 0; k < nk; ++k) {
        /* bottom-left */
        f[IDX(0,0,k)] = f[IDX(0,5,k)];
        f[IDX(0,1,k)] = f[IDX(1,5,k)];
        f[IDX(0,2,k)] = f[IDX(2,5,k)];
        f[IDX(1,0,k)] = f[IDX(0,4,k)];
        f[IDX(1,1,k)] = f[IDX(1,4,k)];
        f[IDX(1,2,k)] = f[IDX(2,4,k)];
        f[IDX(2,0,k)] = f[IDX(0,3,k)];
        f[IDX(2,1,k)] = f[IDX(1,3,k)];
        f[IDX(2,2,k)] = f[IDX(2,3,k)];
        /* bottom-right */
        f[IDX(0,ny-4,k)] = f[IDX(2,ny-7,k)];
        f[IDX(0,ny-3,k)] = f[IDX(1,ny-7,k)];
        f[IDX(0,ny-2,k)] = f[IDX(0,ny-7,k)];
        f[IDX(1,ny-4,k)] = f[IDX(2,ny-6,k)];
        f[IDX(1,ny-3,k)] = f[IDX(1,ny-6,k)];
        f[IDX(1,ny-2,k)] = f[IDX(0,ny-6,k)];
        f[IDX(2,ny-4,k)] = f[IDX(2,ny-5,k)];
        f[IDX(2,ny-3,k)] = f[IDX(1,ny-5,k)];
        f[IDX(2,ny-2,k)] = f[IDX(0,ny-5,k)];
        /* top-left */
        f[IDX(nx-4,0,k)] = f[IDX(nx-2,3,k)];
        f[IDX(nx-4,1,k)] = f[IDX(nx-3,3,k)];
        f[IDX(nx-4,2,k)] = f[IDX(nx-4,3,k)];
        f[IDX(nx-3,0,k)] = f[IDX(nx-2,4,k)];
        f[IDX(nx-3,1,k)] = f[IDX(nx-3,4,k)];
        f[IDX(nx-3,2,k)] = f[IDX(nx-4,4,k)];
        f[IDX(nx-2,0,k)] = f[IDX(nx-2,5,k)];
        f[IDX(nx-2,1,k)] = f[IDX(nx-3,5,k)];
        f[IDX(nx-2,2,k)] = f[IDX(nx-4,5,k)];
        /* top-right */
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

    for (int64_t i = lo; i < hi; ++i) {
        for (int64_t j = 0; j < ny; ++j) {
            for (int64_t k = 0; k < nk; ++k) {
                al[IDX(i,j,k)] = P1 * (q[IDX(i-1,j,k)] + q[IDX(i,j,k)]) +
                                 P2 * (q[IDX(i-2,j,k)] + q[IDX(i+1,j,k)]);
            }
        }
    }

    if (grid_type < 3) {
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

    for (int64_t j = lo; j < hi; ++j) {
        for (int64_t i = 0; i < nx; ++i) {
            for (int64_t k = 0; k < nk; ++k) {
                al[IDX(i,j,k)] = P1 * (q[IDX(i,j-1,k)] + q[IDX(i,j,k)]) +
                                 P2 * (q[IDX(i,j-2,k)] + q[IDX(i,j+1,k)]);
            }
        }
    }

    if (grid_type < 3) {
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

static void xppm_flux(const double * restrict q, const double * restrict courant,
                      const double * restrict al, double * restrict xflux,
                      int64_t nhalo, int64_t ni, int64_t nk,
                      int64_t nx, int64_t ny, int64_t mord)
{
    int64_t i_start = nhalo;
    int64_t i_end   = nhalo + ni - 1;
    int64_t lo = i_start;
    int64_t hi = i_end + 2;

    for (int64_t i = lo; i < hi; ++i) {
        for (int64_t j = 0; j < ny; ++j) {
            for (int64_t k = 0; k < nk; ++k) {
                double qi  = q[IDX(i,j,k)];
                double qim = q[IDX(i-1,j,k)];
                double bl  = al[IDX(i,j,k)]   - qi;
                double br  = al[IDX(i+1,j,k)] - qi;
                double b0  = bl + br;
                double blm = al[IDX(i-1,j,k)] - qim;
                double brm = al[IDX(i,j,k)]   - qim;
                double b0m = blm + brm;

                int smt5, smt5m;
                if (mord == 5) {
                    smt5  = (bl * br < 0.0);
                    smt5m = (blm * brm < 0.0);
                } else {
                    smt5  = (3.0 * fabs(b0) < fabs(bl - br));
                    smt5m = (3.0 * fabs(b0m) < fabs(blm - brm));
                }
                double mask = (smt5 || smt5m) ? 1.0 : 0.0;

                double c = courant[IDX(i,j,k)];
                if (c > 0.0) {
                    xflux[IDX(i,j,k)] = qim + (1.0 - c) * (brm - c * b0m) * mask;
                } else {
                    xflux[IDX(i,j,k)] = qi  + (1.0 + c) * (bl  + c * b0)  * mask;
                }
                int64_t idxp = IDX(i,j,k);
                if (idxp == 478) {
                    fprintf(stderr, "xppm i=%ld j=%ld k=%ld mord=%ld bl=%.17g br=%.17g blm=%.17g brm=%.17g smt5=%d smt5m=%d mask=%.17g c=%.17g qi=%.17g out=%.17g\n",
                            (long)i,(long)j,(long)k,(long)mord,bl,br,blm,brm,smt5,smt5m,mask,c,qi,xflux[idxp]);
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

    for (int64_t j = lo; j < hi; ++j) {
        for (int64_t i = 0; i < nx; ++i) {
            for (int64_t k = 0; k < nk; ++k) {
                double qj  = q[IDX(i,j,k)];
                double qjm = q[IDX(i,j-1,k)];
                double bl  = al[IDX(i,j,k)]   - qj;
                double br  = al[IDX(i,j+1,k)] - qj;
                double b0  = bl + br;
                double blm = al[IDX(i,j-1,k)] - qjm;
                double brm = al[IDX(i,j,k)]   - qjm;
                double b0m = blm + brm;

                int smt5, smt5m;
                if (mord == 5) {
                    smt5  = (bl * br < 0.0);
                    smt5m = (blm * brm < 0.0);
                } else {
                    smt5  = (3.0 * fabs(b0) < fabs(bl - br));
                    smt5m = (3.0 * fabs(b0m) < fabs(blm - brm));
                }
                double mask = (smt5 || smt5m) ? 1.0 : 0.0;

                double c = courant[IDX(i,j,k)];
                if (c > 0.0) {
                    yflux[IDX(i,j,k)] = qjm + (1.0 - c) * (brm - c * b0m) * mask;
                } else {
                    yflux[IDX(i,j,k)] = qj  + (1.0 + c) * (bl  + c * b0)  * mask;
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
    for (int64_t j = j0; j < j1; ++j) {
        for (int64_t i = 0; i < nx; ++i) {
            for (int64_t k = 0; k < nk; ++k) {
                double fyy_j   = y_area_flux[IDX(i,j,k)]   * q_advected_along_y[IDX(i,j,k)];
                double fyy_jp1 = y_area_flux[IDX(i,j+1,k)] * q_advected_along_y[IDX(i,j+1,k)];
                double denom = area[IDX(i,j,k)] + y_area_flux[IDX(i,j,k)] - y_area_flux[IDX(i,j+1,k)];
                q_i[IDX(i,j,k)] = (q[IDX(i,j,k)] * area[IDX(i,j,k)] + fyy_j - fyy_jp1) / denom;
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
    for (int64_t i = i0; i < i1; ++i) {
        for (int64_t j = 0; j < ny; ++j) {
            for (int64_t k = 0; k < nk; ++k) {
                double fx1_i   = x_area_flux[IDX(i,j,k)]   * fx2[IDX(i,j,k)];
                double fx1_ip1 = x_area_flux[IDX(i+1,j,k)] * fx2[IDX(i+1,j,k)];
                double denom = area[IDX(i,j,k)] + x_area_flux[IDX(i,j,k)] - x_area_flux[IDX(i+1,j,k)];
                q_j[IDX(i,j,k)] = (q[IDX(i,j,k)] * area[IDX(i,j,k)] + fx1_i - fx1_ip1) / denom;
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

    for (int64_t i = i_start; i < i_end + 2; ++i) {
        for (int64_t j = j_start; j < j_end + 1; ++j) {
            for (int64_t k = 0; k < nk; ++k) {
                x_flux[IDX(i,j,k)] = 0.5 * (q_ayxa[IDX(i,j,k)] + q_xa[IDX(i,j,k)])
                                         * x_unit_flux[IDX(i,j,k)];
            }
        }
    }
    for (int64_t i = i_start; i < i_end + 1; ++i) {
        for (int64_t j = j_start; j < j_end + 2; ++j) {
            for (int64_t k = 0; k < nk; ++k) {
                y_flux[IDX(i,j,k)] = 0.5 * (q_axya[IDX(i,j,k)] + q_ya[IDX(i,j,k)])
                                         * y_unit_flux[IDX(i,j,k)];
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
                               uint8_t * restrict workspace, int64_t workspace_bytes)
{
    const int64_t nhalo = NH;
    int64_t nx = nhalo + ni + nhalo;
    int64_t ny = nhalo + nj + nhalo;
    int64_t nxyz = nx * ny * nk;
    size_t need = (size_t)nxyz * sizeof(double);
    size_t total_need = 7 * need;

    double * restrict ws;
    int use_workspace = (workspace && (int64_t)total_need <= workspace_bytes);
    if (use_workspace) {
        ws = (double *)workspace;
    } else {
        ws = (double *)malloc(total_need);
    }

    double * restrict q_y_advected_mean = ws + 0 * nxyz;
    double * restrict q_x_advected_mean = ws + 1 * nxyz;
    double * restrict q_advected_y      = ws + 2 * nxyz;
    double * restrict q_advected_x      = ws + 3 * nxyz;
    double * restrict q_ayxa           = ws + 4 * nxyz;
    double * restrict q_axya           = ws + 5 * nxyz;
    double * restrict al               = ws + 6 * nxyz;
    memset(ws, 0, total_need);

    int64_t ord_outer = hord;
    int64_t ord_inner = (hord == 10) ? 8 : hord;

    /* Path 1: y advection then x advection */
    copy_corners_y(q, nx, ny, nk);
    compute_al_y(q, dya, al, nhalo, nj, nk, nx, ny, grid_type);
    yppm_flux(q, cry, al, q_y_advected_mean, nhalo, nj, nk, nx, ny, ord_inner);
    q_i_stencil(q, area, y_area_flux, q_y_advected_mean, q_advected_y, nk, nx, ny);
    compute_al_x(q_advected_y, dxa, al, nhalo, ni, nk, nx, ny, grid_type);
    xppm_flux(q_advected_y, crx, al, q_ayxa, nhalo, ni, nk, nx, ny, ord_outer);

    /* Path 2: x advection then y advection */
    copy_corners_x(q, nx, ny, nk);
    compute_al_x(q, dxa, al, nhalo, ni, nk, nx, ny, grid_type);
    xppm_flux(q, crx, al, q_x_advected_mean, nhalo, ni, nk, nx, ny, ord_inner);
    q_j_stencil(q, area, x_area_flux, q_x_advected_mean, q_advected_x, nk, nx, ny);
    compute_al_y(q_advected_x, dya, al, nhalo, nj, nk, nx, ny, grid_type);
    yppm_flux(q_advected_x, cry, al, q_axya, nhalo, nj, nk, nx, ny, ord_outer);

    {
        FILE *f = fopen("/shared/agent-19/dbg_qxa.bin","wb");
        fwrite(q_x_advected_mean, sizeof(double), (size_t)nxyz, f); fclose(f);
        f = fopen("/shared/agent-19/dbg_qayxa.bin","wb");
        fwrite(q_ayxa, sizeof(double), (size_t)nxyz, f); fclose(f);
        f = fopen("/shared/agent-19/dbg_qya.bin","wb");
        fwrite(q_y_advected_mean, sizeof(double), (size_t)nxyz, f); fclose(f);
        f = fopen("/shared/agent-19/dbg_qaxya.bin","wb");
        fwrite(q_axya, sizeof(double), (size_t)nxyz, f); fclose(f);
    }

    final_fluxes(q_ayxa, q_x_advected_mean, q_axya, q_y_advected_mean,
                 x_area_flux, y_area_flux, q_x_flux, q_y_flux,
                 nhalo, ni, nj, nk, nx, ny);

    if (!use_workspace) free(ws);
}
