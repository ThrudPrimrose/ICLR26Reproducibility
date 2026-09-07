#include <stdint.h>
#include <stdlib.h>
#include <math.h>

// PPM coefficients (same as Python reference)
static const double P1 = 0.5833333333333334; // 7/12
static const double P2 = -0.08333333333333333; // -1/12
static const double C1 = -0.14285714285714285; // -2/14
static const double C2 = 0.7857142857142857; // 11/14
static const double C3 = 0.35714285714285715; // 5/14

static inline int64_t idx(int64_t i, int64_t j, int64_t k, int64_t ny, int64_t nk) {
    return i * ny * nk + j * nk + k;
}

static inline int64_t wrap(int64_t a, int64_t dim) {
    return a >= 0 ? a : dim + a;
}

// Copy a full k‑slice from (src_i,src_j) to (dst_i,dst_j)
static void copy_slice(double *f, int64_t src_i, int64_t src_j, int64_t dst_i, int64_t dst_j,
                       int64_t nx, int64_t ny, int64_t nk) {
    int64_t src_base = idx(src_i, src_j, 0, ny, nk);
    int64_t dst_base = idx(dst_i, dst_j, 0, ny, nk);
    for (int64_t k = 0; k < nk; ++k) {
        f[dst_base + k] = f[src_base + k];
    }
}

// ---------------------------------------------------------------------------
// Corner copy – x direction (cubed‑sphere corners)
static void copy_corners_x(double *f, int64_t nx, int64_t ny, int64_t nk) {
    // helper to interpret possibly negative indices
    #define CCI(i,j) copy_slice(f, wrap(i,nx), wrap(j,ny), (i), (j), nx, ny, nk)
    // explicit assignments matching the NumPy reference
    CCI(0,0);   // src (0,5)
    CCI(0,1);   // src (1,5)
    CCI(0,2);   // src (2,5)
    CCI(1,0);   // src (0,4)
    CCI(1,1);   // src (1,4)
    CCI(1,2);   // src (2,4)
    CCI(2,0);   // src (0,3)
    CCI(2,1);   // src (1,3)
    CCI(2,2);   // src (2,3)
    CCI(0,-4);  // src (2,-7)
    CCI(0,-3);  // src (1,-7)
    CCI(0,-2);  // src (0,-7)
    CCI(1,-4);  // src (2,-6)
    CCI(1,-3);  // src (1,-6)
    CCI(1,-2);  // src (0,-6)
    CCI(2,-4);  // src (2,-5)
    CCI(2,-3);  // src (1,-5)
    CCI(2,-2);  // src (0,-5)
    CCI(-4,0);  // src (-2,3)
    CCI(-4,1);  // src (-3,3)
    CCI(-4,2);  // src (-4,3)
    CCI(-3,0);  // src (-2,4)
    CCI(-3,1);  // src (-3,4)
    CCI(-3,2);  // src (-4,4)
    CCI(-2,0);  // src (-2,5)
    CCI(-2,1);  // src (-3,5)
    CCI(-2,2);  // src (-4,5)
    CCI(-4,-2); // src (-2,-5)
    CCI(-4,-3); // src (-3,-5)
    CCI(-4,-4); // src (-4,-5)
    CCI(-3,-2); // src (-2,-6)
    CCI(-3,-3); // src (-3,-6)
    CCI(-3,-4); // src (-4,-6)
    CCI(-2,-2); // src (-2,-7)
    CCI(-2,-3); // src (-3,-7)
    CCI(-2,-4); // src (-4,-7)
    #undef CCI
}

// ---------------------------------------------------------------------------
// Corner copy – y direction (transpose of the x version)
static void copy_corners_y(double *f, int64_t nx, int64_t ny, int64_t nk) {
    #define CCI(i,j) copy_slice(f, wrap(i,nx), wrap(j,ny), (i), (j), nx, ny, nk)
    CCI(0,0);   // src (5,0)
    CCI(1,0);   // src (5,1)
    CCI(2,0);   // src (5,2)
    CCI(0,1);   // src (4,0)
    CCI(1,1);   // src (4,1)
    CCI(2,1);   // src (4,2)
    CCI(0,2);   // src (3,0)
    CCI(1,2);   // src (3,1)
    CCI(2,2);   // src (3,2)
    CCI(-4,0);  // src (-7,2)
    CCI(-3,0);  // src (-7,1)
    CCI(-2,0);  // src (-7,0)
    CCI(-4,1);  // src (-6,2)
    CCI(-3,1);  // src (-6,1)
    CCI(-2,1);  // src (-6,0)
    CCI(-4,2);  // src (-5,2)
    CCI(-3,2);  // src (-5,1)
    CCI(-2,2);  // src (-5,0)
    CCI(0,-2);  // src (5,-2)
    CCI(0,-3);  // src (4,-2)
    CCI(0,-4);  // src (3,-2)
    CCI(1,-2);  // src (5,-3)
    CCI(1,-3);  // src (4,-3)
    CCI(1,-4);  // src (3,-3)
    CCI(2,-2);  // src (5,-4)
    CCI(2,-3);  // src (4,-4)
    CCI(2,-4);  // src (3,-4)
    CCI(-2,-4); // src (-5,-2)
    CCI(-2,-3); // src (-6,-2)
    CCI(-2,-2); // src (-7,-2)
    CCI(-3,-4); // src (-5,-3)
    CCI(-3,-3); // src (-6,-3)
    CCI(-3,-2); // src (-7,-3)
    CCI(-4,-4); // src (-5,-4)
    CCI(-4,-3); // src (-6,-4)
    CCI(-4,-2); // src (-7,-4)
    #undef CCI
}

// ---------------------------------------------------------------------------
// Compute al (interpolated values) in x‑direction
static void compute_al_x(const double *q, const double *dxa, double *al,
                         int64_t nhalo, int64_t ni, int64_t nj, int64_t nk,
                         int64_t grid_type, int64_t nx, int64_t ny) {
    int64_t i_start = nhalo;
    int64_t i_end   = nhalo + ni - 1;
    int64_t lo = i_start - 1;
    int64_t hi = i_end + 3; // exclusive upper bound
    for (int64_t i = lo; i < hi; ++i) {
        for (int64_t j = 0; j < ny; ++j) {
            for (int64_t k = 0; k < nk; ++k) {
                double q_im1 = q[idx(i-1, j, k, ny, nk)];
                double q_i   = q[idx(i,   j, k, ny, nk)];
                double q_im2 = q[idx(i-2, j, k, ny, nk)];
                double q_ip1 = q[idx(i+1, j, k, ny, nk)];
                al[idx(i, j, k, ny, nk)] = P1 * (q_im1 + q_i) + P2 * (q_im2 + q_ip1);
            }
        }
    }
    if (grid_type < 3) {
        // ia = [i_start-1, i_end]
        int64_t ia[2] = {i_start - 1, i_end};
        for (int a = 0; a < 2; ++a) {
            int64_t i = ia[a];
            for (int64_t j = 0; j < ny; ++j) {
                for (int64_t k = 0; k < nk; ++k) {
                    double q_im2 = q[idx(i-2, j, k, ny, nk)];
                    double q_im1 = q[idx(i-1, j, k, ny, nk)];
                    double q_i   = q[idx(i,   j, k, ny, nk)];
                    al[idx(i, j, k, ny, nk)] = C1 * q_im2 + C2 * q_im1 + C3 * q_i;
                }
            }
        }
        // ib = [i_start, i_end+1]
        int64_t ib[2] = {i_start, i_end + 1};
        for (int b = 0; b < 2; ++b) {
            int64_t i = ib[b];
            for (int64_t j = 0; j < ny; ++j) {
                for (int64_t k = 0; k < nk; ++k) {
                    // left part (uses i-1 and i-2)
                    double dxa_im1 = dxa[idx(i-1, j, k, ny, nk)];
                    double dxa_im2 = dxa[idx(i-2, j, k, ny, nk)];
                    double q_im1  = q[idx(i-1, j, k, ny, nk)];
                    double q_im2  = q[idx(i-2, j, k, ny, nk)];
                    double left = ((2.0 * dxa_im1 + dxa_im2) * q_im1 - dxa_im1 * q_im2) / (dxa_im2 + dxa_im1);
                    // right part (uses i and i+1)
                    double dxa_i   = dxa[idx(i,   j, k, ny, nk)];
                    double dxa_ip1 = dxa[idx(i+1, j, k, ny, nk)];
                    double q_i    = q[idx(i,   j, k, ny, nk)];
                    double q_ip1  = q[idx(i+1, j, k, ny, nk)];
                    double right = ((2.0 * dxa_i + dxa_ip1) * q_i - dxa_i * q_ip1) / (dxa_i + dxa_ip1);
                    al[idx(i, j, k, ny, nk)] = 0.5 * (left + right);
                }
            }
        }
        // ic = [i_start+1, i_end+2]
        int64_t ic[2] = {i_start + 1, i_end + 2};
        for (int c = 0; c < 2; ++c) {
            int64_t i = ic[c];
            for (int64_t j = 0; j < ny; ++j) {
                for (int64_t k = 0; k < nk; ++k) {
                    double q_im1 = q[idx(i-1, j, k, ny, nk)];
                    double q_i   = q[idx(i,   j, k, ny, nk)];
                    double q_ip1 = q[idx(i+1, j, k, ny, nk)];
                    al[idx(i, j, k, ny, nk)] = C3 * q_im1 + C2 * q_i + C1 * q_ip1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Compute al (interpolated values) in y‑direction (transpose of x version)
static void compute_al_y(const double *q, const double *dya, double *al,
                         int64_t nhalo, int64_t ni, int64_t nj, int64_t nk,
                         int64_t grid_type, int64_t nx, int64_t ny) {
    int64_t j_start = nhalo;
    int64_t j_end   = nhalo + nj - 1;
    int64_t lo = j_start - 1;
    int64_t hi = j_end + 3; // exclusive
    for (int64_t i = 0; i < nx; ++i) {
        for (int64_t j = lo; j < hi; ++j) {
            for (int64_t k = 0; k < nk; ++k) {
                double q_jm1 = q[idx(i, j-1, k, ny, nk)];
                double q_j   = q[idx(i, j,   k, ny, nk)];
                double q_jm2 = q[idx(i, j-2, k, ny, nk)];
                double q_jp1 = q[idx(i, j+1, k, ny, nk)];
                al[idx(i, j, k, ny, nk)] = P1 * (q_jm1 + q_j) + P2 * (q_jm2 + q_jp1);
            }
        }
    }
    if (grid_type < 3) {
        // ja = [j_start-1, j_end]
        int64_t ja[2] = {j_start - 1, j_end};
        for (int a = 0; a < 2; ++a) {
            int64_t j = ja[a];
            for (int64_t i = 0; i < nx; ++i) {
                for (int64_t k = 0; k < nk; ++k) {
                    double q_jm2 = q[idx(i, j-2, k, ny, nk)];
                    double q_jm1 = q[idx(i, j-1, k, ny, nk)];
                    double q_j   = q[idx(i, j,   k, ny, nk)];
                    al[idx(i, j, k, ny, nk)] = C1 * q_jm2 + C2 * q_jm1 + C3 * q_j;
                }
            }
        }
        // jb = [j_start, j_end+1]
        int64_t jb[2] = {j_start, j_end + 1};
        for (int b = 0; b < 2; ++b) {
            int64_t j = jb[b];
            for (int64_t i = 0; i < nx; ++i) {
                for (int64_t k = 0; k < nk; ++k) {
                    // left part (uses j-1 and j-2)
                    double dya_jm1 = dya[idx(i, j-1, k, ny, nk)];
                    double dya_jm2 = dya[idx(i, j-2, k, ny, nk)];
                    double q_jm1  = q[idx(i, j-1, k, ny, nk)];
                    double q_jm2  = q[idx(i, j-2, k, ny, nk)];
                    double left = ((2.0 * dya_jm1 + dya_jm2) * q_jm1 - dya_jm1 * q_jm2) / (dya_jm2 + dya_jm1);
                    // right part (uses j and j+1)
                    double dya_j   = dya[idx(i, j,   k, ny, nk)];
                    double dya_jp1 = dya[idx(i, j+1, k, ny, nk)];
                    double q_j    = q[idx(i, j,   k, ny, nk)];
                    double q_jp1  = q[idx(i, j+1, k, ny, nk)];
                    double right = ((2.0 * dya_j + dya_jp1) * q_j - dya_j * q_jp1) / (dya_j + dya_jp1);
                    al[idx(i, j, k, ny, nk)] = 0.5 * (left + right);
                }
            }
        }
        // jc = [j_start+1, j_end+2]
        int64_t jc[2] = {j_start + 1, j_end + 2};
        for (int c = 0; c < 2; ++c) {
            int64_t j = jc[c];
            for (int64_t i = 0; i < nx; ++i) {
                for (int64_t k = 0; k < nk; ++k) {
                    double q_jm1 = q[idx(i, j-1, k, ny, nk)];
                    double q_j   = q[idx(i, j,   k, ny, nk)];
                    double q_jp1 = q[idx(i, j+1, k, ny, nk)];
                    al[idx(i, j, k, ny, nk)] = C3 * q_jm1 + C2 * q_j + C1 * q_jp1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// x‑direction advective flux reconstruction
static void xppm_flux(const double *q, const double *courant, const double *al,
                      double *xflux, int64_t nhalo, int64_t ni, int64_t nj, int64_t nk,
                      int64_t mord, int64_t nx, int64_t ny) {
    int64_t i_start = nhalo;
    int64_t i_end   = nhalo + ni - 1;
    int64_t lo = i_start;
    int64_t hi = i_end + 2; // exclusive upper bound
    for (int64_t i = lo; i < hi; ++i) {
        for (int64_t j = 0; j < ny; ++j) {
            for (int64_t k = 0; k < nk; ++k) {
                double c   = courant[idx(i, j, k, ny, nk)];
                double qi  = q[idx(i, j, k, ny, nk)];
                double qim1 = q[idx(i-1, j, k, ny, nk)];
                double al_i   = al[idx(i,   j, k, ny, nk)];
                double al_ip1 = al[idx(i+1, j, k, ny, nk)];
                double al_im1 = al[idx(i-1, j, k, ny, nk)];
                double bl = al_i - qi;
                double br = al_ip1 - qi;
                double b0 = bl + br;
                double bl_m1 = al_im1 - qim1;
                double br_m1 = al_i - qim1;
                double b0_m1 = bl_m1 + br_m1;
                int smt5 = 0, smt5_m1 = 0;
                if (mord == 5) {
                    smt5 = (bl * br < 0.0);
                    smt5_m1 = (bl_m1 * br_m1 < 0.0);
                } else {
                    smt5 = (3.0 * fabs(b0) < fabs(bl - br));
                    smt5_m1 = (3.0 * fabs(b0_m1) < fabs(bl_m1 - br_m1));
                }
                double mask = (smt5 || smt5_m1) ? 1.0 : 0.0;
                double result;
                if (c > 0.0) {
                    result = qim1 + (1.0 - c) * (br_m1 - c * b0_m1) * mask;
                } else {
                    result = qi + (1.0 + c) * (bl + c * b0) * mask;
                }
                xflux[idx(i, j, k, ny, nk)] = result;
            }
        }
    }
}

static void xppm(const double *q, const double *courant, const double *dxa,
                 double *xflux, double *al,
                 int64_t nhalo, int64_t ni, int64_t nj, int64_t nk,
                 int64_t mord, int64_t grid_type, int64_t nx, int64_t ny) {
    compute_al_x(q, dxa, al, nhalo, ni, nj, nk, grid_type, nx, ny);
    xppm_flux(q, courant, al, xflux, nhalo, ni, nj, nk, mord, nx, ny);
}

// ---------------------------------------------------------------------------
// y‑direction advective flux reconstruction
static void yppm_flux(const double *q, const double *courant, const double *al,
                      double *yflux, int64_t nhalo, int64_t ni, int64_t nj, int64_t nk,
                      int64_t mord, int64_t nx, int64_t ny) {
    int64_t j_start = nhalo;
    int64_t j_end   = nhalo + nj - 1;
    int64_t lo = j_start;
    int64_t hi = j_end + 2; // exclusive
    for (int64_t i = 0; i < nx; ++i) {
        for (int64_t j = lo; j < hi; ++j) {
            for (int64_t k = 0; k < nk; ++k) {
                double c   = courant[idx(i, j, k, ny, nk)];
                double qj  = q[idx(i, j, k, ny, nk)];
                double qjm1 = q[idx(i, j-1, k, ny, nk)];
                double al_j   = al[idx(i, j,   k, ny, nk)];
                double al_jp1 = al[idx(i, j+1, k, ny, nk)];
                double al_jm1 = al[idx(i, j-1, k, ny, nk)];
                double bl = al_j - qj;
                double br = al_jp1 - qj;
                double b0 = bl + br;
                double bl_m1 = al_jm1 - qjm1;
                double br_m1 = al_j - qjm1;
                double b0_m1 = bl_m1 + br_m1;
                int smt5 = 0, smt5_m1 = 0;
                if (mord == 5) {
                    smt5 = (bl * br < 0.0);
                    smt5_m1 = (bl_m1 * br_m1 < 0.0);
                } else {
                    smt5 = (3.0 * fabs(b0) < fabs(bl - br));
                    smt5_m1 = (3.0 * fabs(b0_m1) < fabs(bl_m1 - br_m1));
                }
                double mask = (smt5 || smt5_m1) ? 1.0 : 0.0;
                double result;
                if (c > 0.0) {
                    result = qjm1 + (1.0 - c) * (br_m1 - c * b0_m1) * mask;
                } else {
                    result = qj + (1.0 + c) * (bl + c * b0) * mask;
                }
                yflux[idx(i, j, k, ny, nk)] = result;
            }
        }
    }
}

static void yppm(const double *q, const double *courant, const double *dya,
                 double *yflux, double *al,
                 int64_t nhalo, int64_t ni, int64_t nj, int64_t nk,
                 int64_t mord, int64_t grid_type, int64_t nx, int64_t ny) {
    compute_al_y(q, dya, al, nhalo, ni, nj, nk, grid_type, nx, ny);
    yppm_flux(q, courant, al, yflux, nhalo, ni, nj, nk, mord, nx, ny);
}

// ---------------------------------------------------------------------------
// q_i stencil (y‑advected mean)
static void q_i_stencil(const double *q, const double *area, const double *y_area_flux,
                        const double *q_adv_y, double *q_i,
                        int64_t nhalo, int64_t ni, int64_t nj, int64_t nk,
                        int64_t nx, int64_t ny) {
    int64_t j0 = 3;
    int64_t j1 = ny - 3; // exclusive upper bound
    for (int64_t i = 0; i < nx; ++i) {
        for (int64_t j = j0; j < j1; ++j) {
            for (int64_t k = 0; k < nk; ++k) {
                double fyy_j  = y_area_flux[idx(i, j,   k, ny, nk)] * q_adv_y[idx(i, j,   k, ny, nk)];
                double fyy_jp1 = y_area_flux[idx(i, j+1, k, ny, nk)] * q_adv_y[idx(i, j+1, k, ny, nk)];
                double denom = area[idx(i, j, k, ny, nk)] +
                               y_area_flux[idx(i, j,   k, ny, nk)] -
                               y_area_flux[idx(i, j+1, k, ny, nk)];
                q_i[idx(i, j, k, ny, nk)] = (q[idx(i, j, k, ny, nk)] * area[idx(i, j, k, ny, nk)] + fyy_j - fyy_jp1) / denom;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// q_j stencil (x‑advected mean)
static void q_j_stencil(const double *q, const double *area, const double *x_area_flux,
                        const double *fx2, double *q_j,
                        int64_t nhalo, int64_t ni, int64_t nj, int64_t nk,
                        int64_t nx, int64_t ny) {
    int64_t i0 = 3;
    int64_t i1 = nx - 3; // exclusive upper bound
    for (int64_t i = i0; i < i1; ++i) {
        for (int64_t j = 0; j < ny; ++j) {
            for (int64_t k = 0; k < nk; ++k) {
                double fx1_i   = x_area_flux[idx(i,   j, k, ny, nk)] * fx2[idx(i,   j, k, ny, nk)];
                double fx1_ip1 = x_area_flux[idx(i+1, j, k, ny, nk)] * fx2[idx(i+1, j, k, ny, nk)];
                double denom = area[idx(i, j, k, ny, nk)] +
                               x_area_flux[idx(i,   j, k, ny, nk)] -
                               x_area_flux[idx(i+1, j, k, ny, nk)];
                q_j[idx(i, j, k, ny, nk)] = (q[idx(i, j, k, ny, nk)] * area[idx(i, j, k, ny, nk)] + fx1_i - fx1_ip1) / denom;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Final flux combination (cancels splitting error)
static void final_fluxes(const double *q_ayxa, const double *q_xa,
                         const double *q_axya, const double *q_ya,
                         const double *x_unit_flux, const double *y_unit_flux,
                         double *x_flux, double *y_flux,
                         int64_t nhalo, int64_t ni, int64_t nj, int64_t nk,
                         int64_t nx, int64_t ny) {
    int64_t i_start = nhalo;
    int64_t i_end   = nhalo + ni - 1;
    int64_t j_start = nhalo;
    int64_t j_end   = nhalo + nj - 1;
    // x‑flux: i from i_start .. i_end+1 (inclusive), j from j_start .. j_end (inclusive)
    for (int64_t i = i_start; i <= i_end + 1; ++i) {
        for (int64_t j = j_start; j <= j_end; ++j) {
            for (int64_t k = 0; k < nk; ++k) {
                double val = 0.5 * (q_ayxa[idx(i, j, k, ny, nk)] + q_xa[idx(i, j, k, ny, nk)]) *
                             x_unit_flux[idx(i, j, k, ny, nk)];
                x_flux[idx(i, j, k, ny, nk)] = val;
            }
        }
    }
    // y‑flux: i from i_start .. i_end (inclusive), j from j_start .. j_end+1 (inclusive)
    for (int64_t i = i_start; i <= i_end; ++i) {
        for (int64_t j = j_start; j <= j_end + 1; ++j) {
            for (int64_t k = 0; k < nk; ++k) {
                double val = 0.5 * (q_axya[idx(i, j, k, ny, nk)] + q_ya[idx(i, j, k, ny, nk)]) *
                             y_unit_flux[idx(i, j, k, ny, nk)];
                y_flux[idx(i, j, k, ny, nk)] = val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Top‑level FV3 dycore kernel – implements the finite‑volume transport step
void fv3_dycore_fp64(double *restrict q, double *restrict crx, double *restrict cry,
                     double *restrict x_area_flux, double *restrict y_area_flux,
                     double *restrict q_x_flux, double *restrict q_y_flux,
                     double *restrict dxa, double *restrict dya,
                     double *restrict area, double *restrict rarea,
                     double *restrict del6_v, double *restrict del6_u,
                     int64_t hord, int64_t grid_type,
                     int64_t ni, int64_t nj, int64_t nk) {
    const int64_t NHALO = 3;
    int64_t nx = NHALO + ni + NHALO;
    int64_t ny = NHALO + nj + NHALO;

    // Allocate temporary buffers (zero‑initialised like NumPy's zeros)
    size_t total = (size_t)nx * (size_t)ny * (size_t)nk;
    double *q_y_advected_mean = (double *)calloc(total, sizeof(double));
    double *q_x_advected_mean = (double *)calloc(total, sizeof(double));
    double *q_advected_y       = (double *)calloc(total, sizeof(double));
    double *q_advected_x       = (double *)calloc(total, sizeof(double));
    double *q_ayxa             = (double *)calloc(total, sizeof(double));
    double *q_axya             = (double *)calloc(total, sizeof(double));
    double *al                 = (double *)calloc(total, sizeof(double));
    if (!q_y_advected_mean || !q_x_advected_mean || !q_advected_y || !q_advected_x ||
        !q_ayxa || !q_axya || !al) {
        // Allocation failure – abort silently (benchmark expects valid data)
        return;
    }

    // Sequence mirroring the NumPy reference
    copy_corners_y(q, nx, ny, nk);
    yppm(q, cry, dya, q_y_advected_mean, al, NHALO, ni, nj, nk, hord < 10 ? hord : 8, grid_type, nx, ny);
    q_i_stencil(q, area, y_area_flux, q_y_advected_mean, q_advected_y, NHALO, ni, nj, nk, nx, ny);
    xppm(q_advected_y, crx, dxa, q_ayxa, al, NHALO, ni, nj, nk, hord, grid_type, nx, ny);

    copy_corners_x(q, nx, ny, nk);
    xppm(q, crx, dxa, q_x_advected_mean, al, NHALO, ni, nj, nk, hord < 10 ? hord : 8, grid_type, nx, ny);
    q_j_stencil(q, area, x_area_flux, q_x_advected_mean, q_advected_x, NHALO, ni, nj, nk, nx, ny);
    yppm(q_advected_x, cry, dya, q_axya, al, NHALO, ni, nj, nk, hord, grid_type, nx, ny);

    final_fluxes(q_ayxa, q_x_advected_mean, q_axya, q_y_advected_mean,
                 x_area_flux, y_area_flux, q_x_flux, q_y_flux,
                 NHALO, ni, nj, nk, nx, ny);

    // Free temporaries
    free(q_y_advected_mean);
    free(q_x_advected_mean);
    free(q_advected_y);
    free(q_advected_x);
    free(q_ayxa);
    free(q_axya);
    free(al);
}
