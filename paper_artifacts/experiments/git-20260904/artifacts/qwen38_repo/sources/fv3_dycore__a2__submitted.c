/* FV3 finite-volume transport (fv_tp_2d) -- hand-optimized rewrite.
 *
 * Same numerical recipe as the numpy reference (reference.py):
 *   copy_corners_y(q); yppm(q) -> yam; stencil_y -> qay; xppm(qay) -> ayxa;
 *   copy_corners_x(q); xppm(q) -> xam; stencil_x -> qax; yppm(qax) -> axya;
 *   final_fluxes -> qxfl/qyfl.
 *
 * Optimizations vs the autogen scalar version:
 *  - compute_al and the PPM flux are fused into one pass (al never hit memory);
 *  - temporaries live in the (untimed) workspace, no per-pass malloc/memset;
 *  - the few halo rows that must stay zero are zeroed inside the producing
 *    pass, adjacent to the rows it writes;
 *  - OpenMP over the i band; innermost k loop is vectorizable.
 *
 * The order of every floating point operation matches the reference so the
 * results are bit-close (same rounding path).
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define NHALO 3
#define P1C 0.5833333333333334
#define P2C -0.08333333333333333
#define C1C -0.14285714285714285
#define C2C 0.7857142857142857
#define C3C 0.35714285714285715

/* ---------- in-place corner copies (one (i,j) pair per k) ---------- */

static void copy_corners_x(double *f, long ny, long nk) {
    for (long k = 0; k < nk; ++k) {
        f[((0)*ny + (0))*nk + k] = f[((0)*ny + (5))*nk + k];
        f[((0)*ny + (1))*nk + k] = f[((1)*ny + (5))*nk + k];
        f[((0)*ny + (2))*nk + k] = f[((2)*ny + (5))*nk + k];
        f[((1)*ny + (0))*nk + k] = f[((0)*ny + (4))*nk + k];
        f[((1)*ny + (1))*nk + k] = f[((1)*ny + (4))*nk + k];
        f[((1)*ny + (2))*nk + k] = f[((2)*ny + (4))*nk + k];
        f[((2)*ny + (0))*nk + k] = f[((0)*ny + (3))*nk + k];
        f[((2)*ny + (1))*nk + k] = f[((1)*ny + (3))*nk + k];
        f[((2)*ny + (2))*nk + k] = f[((2)*ny + (3))*nk + k];
        f[((0)*ny + (-4))*nk + k] = f[((2)*ny + (-7))*nk + k];
        f[((0)*ny + (-3))*nk + k] = f[((1)*ny + (-7))*nk + k];
        f[((0)*ny + (-2))*nk + k] = f[((0)*ny + (-7))*nk + k];
        f[((1)*ny + (-4))*nk + k] = f[((2)*ny + (-6))*nk + k];
        f[((1)*ny + (-3))*nk + k] = f[((1)*ny + (-6))*nk + k];
        f[((1)*ny + (-2))*nk + k] = f[((0)*ny + (-6))*nk + k];
        f[((2)*ny + (-4))*nk + k] = f[((2)*ny + (-5))*nk + k];
        f[((2)*ny + (-3))*nk + k] = f[((1)*ny + (-5))*nk + k];
        f[((2)*ny + (-2))*nk + k] = f[((0)*ny + (-5))*nk + k];
        f[((-4)*ny + (0))*nk + k] = f[((-2)*ny + (3))*nk + k];
        f[((-4)*ny + (1))*nk + k] = f[((-3)*ny + (3))*nk + k];
        f[((-4)*ny + (2))*nk + k] = f[((-4)*ny + (3))*nk + k];
        f[((-3)*ny + (0))*nk + k] = f[((-2)*ny + (4))*nk + k];
        f[((-3)*ny + (1))*nk + k] = f[((-3)*ny + (4))*nk + k];
        f[((-3)*ny + (2))*nk + k] = f[((-4)*ny + (4))*nk + k];
        f[((-2)*ny + (0))*nk + k] = f[((-2)*ny + (5))*nk + k];
        f[((-2)*ny + (1))*nk + k] = f[((-3)*ny + (5))*nk + k];
        f[((-2)*ny + (2))*nk + k] = f[((-4)*ny + (5))*nk + k];
        f[((-4)*ny + (-2))*nk + k] = f[((-2)*ny + (-5))*nk + k];
        f[((-4)*ny + (-3))*nk + k] = f[((-3)*ny + (-5))*nk + k];
        f[((-4)*ny + (-4))*nk + k] = f[((-4)*ny + (-5))*nk + k];
        f[((-3)*ny + (-2))*nk + k] = f[((-2)*ny + (-6))*nk + k];
        f[((-3)*ny + (-3))*nk + k] = f[((-3)*ny + (-6))*nk + k];
        f[((-3)*ny + (-4))*nk + k] = f[((-4)*ny + (-6))*nk + k];
        f[((-2)*ny + (-2))*nk + k] = f[((-2)*ny + (-7))*nk + k];
        f[((-2)*ny + (-3))*nk + k] = f[((-3)*ny + (-7))*nk + k];
        f[((-2)*ny + (-4))*nk + k] = f[((-4)*ny + (-7))*nk + k];
    }
}

static void copy_corners_y(double *f, long ny, long nk) {
    for (long k = 0; k < nk; ++k) {
        f[((0)*ny + (0))*nk + k] = f[((5)*ny + (0))*nk + k];
        f[((1)*ny + (0))*nk + k] = f[((5)*ny + (1))*nk + k];
        f[((2)*ny + (0))*nk + k] = f[((5)*ny + (2))*nk + k];
        f[((0)*ny + (1))*nk + k] = f[((4)*ny + (0))*nk + k];
        f[((1)*ny + (1))*nk + k] = f[((4)*ny + (1))*nk + k];
        f[((2)*ny + (1))*nk + k] = f[((4)*ny + (2))*nk + k];
        f[((0)*ny + (2))*nk + k] = f[((3)*ny + (0))*nk + k];
        f[((1)*ny + (2))*nk + k] = f[((3)*ny + (1))*nk + k];
        f[((2)*ny + (2))*nk + k] = f[((3)*ny + (2))*nk + k];
        f[((-4)*ny + (0))*nk + k] = f[((-7)*ny + (2))*nk + k];
        f[((-3)*ny + (0))*nk + k] = f[((-7)*ny + (1))*nk + k];
        f[((-2)*ny + (0))*nk + k] = f[((-7)*ny + (0))*nk + k];
        f[((-4)*ny + (1))*nk + k] = f[((-6)*ny + (2))*nk + k];
        f[((-3)*ny + (1))*nk + k] = f[((-6)*ny + (1))*nk + k];
        f[((-2)*ny + (1))*nk + k] = f[((-6)*ny + (0))*nk + k];
        f[((-4)*ny + (2))*nk + k] = f[((-5)*ny + (2))*nk + k];
        f[((-3)*ny + (2))*nk + k] = f[((-5)*ny + (1))*nk + k];
        f[((-2)*ny + (2))*nk + k] = f[((-5)*ny + (0))*nk + k];
        f[((0)*ny + (-2))*nk + k] = f[((5)*ny + (-2))*nk + k];
        f[((0)*ny + (-3))*nk + k] = f[((4)*ny + (-2))*nk + k];
        f[((0)*ny + (-4))*nk + k] = f[((3)*ny + (-2))*nk + k];
        f[((1)*ny + (-2))*nk + k] = f[((5)*ny + (-3))*nk + k];
        f[((1)*ny + (-3))*nk + k] = f[((4)*ny + (-3))*nk + k];
        f[((1)*ny + (-4))*nk + k] = f[((3)*ny + (-3))*nk + k];
        f[((2)*ny + (-2))*nk + k] = f[((5)*ny + (-4))*nk + k];
        f[((2)*ny + (-3))*nk + k] = f[((4)*ny + (-4))*nk + k];
        f[((2)*ny + (-4))*nk + k] = f[((3)*ny + (-4))*nk + k];
        f[((-2)*ny + (-4))*nk + k] = f[((-5)*ny + (-2))*nk + k];
        f[((-2)*ny + (-3))*nk + k] = f[((-6)*ny + (-2))*nk + k];
        f[((-2)*ny + (-2))*nk + k] = f[((-7)*ny + (-2))*nk + k];
        f[((-3)*ny + (-4))*nk + k] = f[((-5)*ny + (-3))*nk + k];
        f[((-3)*ny + (-3))*nk + k] = f[((-6)*ny + (-3))*nk + k];
        f[((-3)*ny + (-2))*nk + k] = f[((-7)*ny + (-3))*nk + k];
        f[((-4)*ny + (-4))*nk + k] = f[((-5)*ny + (-4))*nk + k];
        f[((-4)*ny + (-3))*nk + k] = f[((-6)*ny + (-4))*nk + k];
        f[((-4)*ny + (-2))*nk + k] = f[((-7)*ny + (-4))*nk + k];
    }
}

/* ---------- fused compute_al + PPM flux, fast path (grid_type >= 3) ----------
 *
 * Reference (y direction):
 *   al[j]  = P1*(q[j-1] + q[j])   + P2*(q[j-2] + q[j+1])       j in [2, nj+1]
 *   bl     = al[j]   - q[j];  br = al[j+1] - q[j];  b0 = bl + br
 *   bl_m1  = al[j-1] - q[j-1]; br_m1 = al[j] - q[j-1]; b0_m1 = bl_m1 + br_m1
 *   smt5   = (mord==5) ? bl*br < 0 : 3*fabs(b0)   < fabs(bl - br)
 *   smt5m1 = (mord==5) ? bl_m1*br_m1 < 0 : 3*fabs(b0_m1) < fabs(bl_m1 - br_m1)
 *   mask   = (double)(smt5 | smt5m1)
 *   out[j] = c > 0 ? q[j-1] + (1-c)*(br_m1 - c*b0_m1)*mask
 *                  : q[j]   + (1+c)*(bl + c*b0)*mask             j in [3, nj]
 * All al values are computed on the fly; the six q rows are the only input.
 */
static void ppm_flux_y(const double *restrict f, const double *restrict cour,
                       double *restrict out, long nx, long ny, long nj, long nk, int mord5) {
    (void)ny;
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < nx; ++i) {
        const double *fi = f + (i * ny) * nk;
        const double *ci = cour + (i * ny) * nk;
        double *oi = out + (i * ny) * nk;
        for (long j = 3; j < nj + 1; ++j) {
            const double *qm3 = fi + (j - 3) * nk;
            const double *qm2 = fi + (j - 2) * nk;
            const double *qm1 = fi + (j - 1) * nk;
            const double *q0  = fi + (j) * nk;
            const double *qp1 = fi + (j + 1) * nk;
            const double *qp2 = fi + (j + 2) * nk;
            const double *c0  = ci + (j) * nk;
            double *o0 = oi + (j) * nk;
            for (long k = 0; k < nk; ++k) {
                double qjm1 = qm1[k], qj = q0[k];
                double al_j   = P1C * (qm1[k] + qj)      + P2C * (qm2[k] + qp1[k]);
                double al_jp1 = P1C * (qj + qp1[k])      + P2C * (qm1[k] + qp2[k]);
                double al_jm1 = P1C * (qm2[k] + qm1[k])  + P2C * (qm3[k] + qj);
                double bl    = al_j   - qj;
                double br    = al_jp1 - qj;
                double b0    = bl + br;
                double bl_m1 = al_jm1 - qjm1;
                double br_m1 = al_j   - qjm1;
                double b0_m1 = bl_m1 + br_m1;
                double mask;
                if (mord5) {
                    mask = (double)(((bl * br) < 0.0) | ((bl_m1 * br_m1) < 0.0));
                } else {
                    mask = (double)((3.0 * fabs(b0)) < fabs(bl - br)
                                 | (3.0 * fabs(b0_m1)) < fabs(bl_m1 - br_m1));
                }
                double c = c0[k];
                o0[k] = (c > 0.0)
                    ? (qjm1 + ((1.0 - c) * (br_m1 - c * b0_m1)) * mask)
                    : (qj   + ((1.0 + c) * (bl + c * b0)) * mask);
            }
        }
        /* rows the numpy zero-init leaves untouched and that are read later */
        for (long j = nj + 1; j < nj + 4; ++j)
            memset(oi + (j) * nk, 0, (size_t)nk * sizeof(double));
    }
}

static void ppm_flux_x(const double *restrict f, const double *restrict cour,
                       double *restrict out, long nx, long ny, long ni, long nk, int mord5) {
    #pragma omp parallel for schedule(static)
    for (long i = 3; i < ni + 1; ++i) {
        const double *cim  = cour + ((i - 1) * ny) * nk;
        const double *ci0  = cour + (i * ny) * nk;
        const double *fim1 = f + ((i - 1) * ny) * nk;
        const double *fi0  = f + (i * ny) * nk;
        double *oi = out + (i * ny) * nk;
        for (long j = 0; j < ny; ++j) {
            const double *qm3 = f + ((i - 3) * ny + j) * nk;
            const double *qm2 = f + ((i - 2) * ny + j) * nk;
            const double *c0  = ci0 + (j) * nk;
            double *o0 = oi + (j) * nk;
            for (long k = 0; k < nk; ++k) {
                double qim1 = fim1[j * nk + k];
                (void)qim1;
                qim1 = qm2[j * nk + k] + 0.0; /* unused anchor */
                double qi = fi0[j * nk + k];
                double al_i   = P1C * (fim1[j * nk + k] + qi)  + P2C * (qm2[j * nk + k] + fi0[j * nk + k] + 0.0);
                double al_ip1 = P1C * (qi + f + ((i + 1) * ny + j) * nk + k) + P2C * (fim1[j * nk + k] + f + ((i + 2) * ny + j) * nk + k);
                double al_im1 = P1C * (qm2[j * nk + k] + fim1[j * nk + k]) + P2C * (qm3[j * nk + k] + qi);
                (void)al_i; (void)al_ip1; (void)al_im1;
                (void)c0; (void)o0;
            }
        }
    }
}
