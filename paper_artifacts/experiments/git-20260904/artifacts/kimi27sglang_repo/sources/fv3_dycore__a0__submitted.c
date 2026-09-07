#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const int64_t NH = 3;

/* Indexing: arrays are (nx, ny, nk) with k contiguous. */
#define IDX3(A, nx, ny, nk, i, j, k) ((A)[(((i) * (ny) + (j)) * (nk) + (k))])

/* PPM reconstruction coefficients. */
static const double P1 = 0.5833333333333334;
static const double P2 = -0.08333333333333333;
static const double C1 = -0.14285714285714285;
static const double C2 = 0.7857142857142857;
static const double C3 = 0.35714285714285715;

static inline int64_t iabs64(int64_t x) { return x < 0 ? -x : x; }

/* ------------------------------------------------------------------ */
/* Copy corners (in-place on q). Parallel over k.                     */
static void copy_corners_y(double *restrict q, int64_t nx, int64_t ny, int64_t nk)
{
    #pragma omp parallel for
    for (int64_t k = 0; k < nk; ++k) {
        q[IDX3(q, nx, ny, nk, 0, 0, k)] = q[IDX3(q, nx, ny, nk, 5, 0, k)];
        q[IDX3(q, nx, ny, nk, 1, 0, k)] = q[IDX3(q, nx, ny, nk, 5, 1, k)];
        q[IDX3(q, nx, ny, nk, 2, 0, k)] = q[IDX3(q, nx, ny, nk, 5, 2, k)];
        q[IDX3(q, nx, ny, nk, 0, 1, k)] = q[IDX3(q, nx, ny, nk, 4, 0, k)];
        q[IDX3(q, nx, ny, nk, 1, 1, k)] = q[IDX3(q, nx, ny, nk, 4, 1, k)];
        q[IDX3(q, nx, ny, nk, 2, 1, k)] = q[IDX3(q, nx, ny, nk, 4, 2, k)];
        q[IDX3(q, nx, ny, nk, 0, 2, k)] = q[IDX3(q, nx, ny, nk, 3, 0, k)];
        q[IDX3(q, nx, ny, nk, 1, 2, k)] = q[IDX3(q, nx, ny, nk, 3, 1, k)];
        q[IDX3(q, nx, ny, nk, 2, 2, k)] = q[IDX3(q, nx, ny, nk, 3, 2, k)];

        q[IDX3(q, nx, ny, nk, nx - 4, 0, k)] = q[IDX3(q, nx, ny, nk, nx - 7, 2, k)];
        q[IDX3(q, nx, ny, nk, nx - 3, 0, k)] = q[IDX3(q, nx, ny, nk, nx - 7, 1, k)];
        q[IDX3(q, nx, ny, nk, nx - 2, 0, k)] = q[IDX3(q, nx, ny, nk, nx - 7, 0, k)];
        q[IDX3(q, nx, ny, nk, nx - 4, 1, k)] = q[IDX3(q, nx, ny, nk, nx - 6, 2, k)];
        q[IDX3(q, nx, ny, nk, nx - 3, 1, k)] = q[IDX3(q, nx, ny, nk, nx - 6, 1, k)];
        q[IDX3(q, nx, ny, nk, nx - 2, 1, k)] = q[IDX3(q, nx, ny, nk, nx - 6, 0, k)];
        q[IDX3(q, nx, ny, nk, nx - 4, 2, k)] = q[IDX3(q, nx, ny, nk, nx - 5, 2, k)];
        q[IDX3(q, nx, ny, nk, nx - 3, 2, k)] = q[IDX3(q, nx, ny, nk, nx - 5, 1, k)];
        q[IDX3(q, nx, ny, nk, nx - 2, 2, k)] = q[IDX3(q, nx, ny, nk, nx - 5, 0, k)];

        q[IDX3(q, nx, ny, nk, 0, ny - 4, k)] = q[IDX3(q, nx, ny, nk, 2, ny - 7, k)];
        q[IDX3(q, nx, ny, nk, 0, ny - 3, k)] = q[IDX3(q, nx, ny, nk, 1, ny - 7, k)];
        q[IDX3(q, nx, ny, nk, 0, ny - 2, k)] = q[IDX3(q, nx, ny, nk, 0, ny - 7, k)];
        q[IDX3(q, nx, ny, nk, 1, ny - 4, k)] = q[IDX3(q, nx, ny, nk, 2, ny - 6, k)];
        q[IDX3(q, nx, ny, nk, 1, ny - 3, k)] = q[IDX3(q, nx, ny, nk, 1, ny - 6, k)];
        q[IDX3(q, nx, ny, nk, 1, ny - 2, k)] = q[IDX3(q, nx, ny, nk, 0, ny - 6, k)];
        q[IDX3(q, nx, ny, nk, 2, ny - 4, k)] = q[IDX3(q, nx, ny, nk, 2, ny - 5, k)];
        q[IDX3(q, nx, ny, nk, 2, ny - 3, k)] = q[IDX3(q, nx, ny, nk, 1, ny - 5, k)];
        q[IDX3(q, nx, ny, nk, 2, ny - 2, k)] = q[IDX3(q, nx, ny, nk, 0, ny - 5, k)];

        q[IDX3(q, nx, ny, nk, nx - 4, ny - 4, k)] = q[IDX3(q, nx, ny, nk, nx - 2, ny - 7, k)];
        q[IDX3(q, nx, ny, nk, nx - 4, ny - 3, k)] = q[IDX3(q, nx, ny, nk, nx - 3, ny - 7, k)];
        q[IDX3(q, nx, ny, nk, nx - 4, ny - 2, k)] = q[IDX3(q, nx, ny, nk, nx - 4, ny - 7, k)];
        q[IDX3(q, nx, ny, nk, nx - 3, ny - 4, k)] = q[IDX3(q, nx, ny, nk, nx - 2, ny - 6, k)];
        q[IDX3(q, nx, ny, nk, nx - 3, ny - 3, k)] = q[IDX3(q, nx, ny, nk, nx - 3, ny - 6, k)];
        q[IDX3(q, nx, ny, nk, nx - 3, ny - 2, k)] = q[IDX3(q, nx, ny, nk, nx - 4, ny - 6, k)];
        q[IDX3(q, nx, ny, nk, nx - 2, ny - 4, k)] = q[IDX3(q, nx, ny, nk, nx - 2, ny - 7, k)];
        q[IDX3(q, nx, ny, nk, nx - 2, ny - 3, k)] = q[IDX3(q, nx, ny, nk, nx - 3, ny - 7, k)];
        q[IDX3(q, nx, ny, nk, nx - 2, ny - 2, k)] = q[IDX3(q, nx, ny, nk, nx - 4, ny - 7, k)];
    }
}

static void copy_corners_x(double *restrict q, int64_t nx, int64_t ny, int64_t nk)
{
    #pragma omp parallel for
    for (int64_t k = 0; k < nk; ++k) {
        q[IDX3(q, nx, ny, nk, 0, 0, k)] = q[IDX3(q, nx, ny, nk, 0, 5, k)];
        q[IDX3(q, nx, ny, nk, 0, 1, k)] = q[IDX3(q, nx, ny, nk, 1, 5, k)];
        q[IDX3(q, nx, ny, nk, 0, 2, k)] = q[IDX3(q, nx, ny, nk, 2, 5, k)];
        q[IDX3(q, nx, ny, nk, 1, 0, k)] = q[IDX3(q, nx, ny, nk, 0, 4, k)];
        q[IDX3(q, nx, ny, nk, 1, 1, k)] = q[IDX3(q, nx, ny, nk, 1, 4, k)];
        q[IDX3(q, nx, ny, nk, 1, 2, k)] = q[IDX3(q, nx, ny, nk, 2, 4, k)];
        q[IDX3(q, nx, ny, nk, 2, 0, k)] = q[IDX3(q, nx, ny, nk, 0, 3, k)];
        q[IDX3(q, nx, ny, nk, 2, 1, k)] = q[IDX3(q, nx, ny, nk, 1, 3, k)];
        q[IDX3(q, nx, ny, nk, 2, 2, k)] = q[IDX3(q, nx, ny, nk, 2, 3, k)];

        q[IDX3(q, nx, ny, nk, 0, ny - 4, k)] = q[IDX3(q, nx, ny, nk, 2, ny - 7, k)];
        q[IDX3(q, nx, ny, nk, 0, ny - 3, k)] = q[IDX3(q, nx, ny, nk, 1, ny - 7, k)];
        q[IDX3(q, nx, ny, nk, 0, ny - 2, k)] = q[IDX3(q, nx, ny, nk, 0, ny - 7, k)];
        q[IDX3(q, nx, ny, nk, 1, ny - 4, k)] = q[IDX3(q, nx, ny, nk, 2, ny - 6, k)];
        q[IDX3(q, nx, ny, nk, 1, ny - 3, k)] = q[IDX3(q, nx, ny, nk, 1, ny - 6, k)];
        q[IDX3(q, nx, ny, nk, 1, ny - 2, k)] = q[IDX3(q, nx, ny, nk, 0, ny - 6, k)];
        q[IDX3(q, nx, ny, nk, 2, ny - 4, k)] = q[IDX3(q, nx, ny, nk, 2, ny - 5, k)];
        q[IDX3(q, nx, ny, nk, 2, ny - 3, k)] = q[IDX3(q, nx, ny, nk, 1, ny - 5, k)];
        q[IDX3(q, nx, ny, nk, 2, ny - 2, k)] = q[IDX3(q, nx, ny, nk, 0, ny - 5, k)];

        q[IDX3(q, nx, ny, nk, nx - 4, 0, k)] = q[IDX3(q, nx, ny, nk, nx - 2, 3, k)];
        q[IDX3(q, nx, ny, nk, nx - 3, 0, k)] = q[IDX3(q, nx, ny, nk, nx - 3, 3, k)];
        q[IDX3(q, nx, ny, nk, nx - 2, 0, k)] = q[IDX3(q, nx, ny, nk, nx - 4, 3, k)];
        q[IDX3(q, nx, ny, nk, nx - 4, 1, k)] = q[IDX3(q, nx, ny, nk, nx - 2, 4, k)];
        q[IDX3(q, nx, ny, nk, nx - 3, 1, k)] = q[IDX3(q, nx, ny, nk, nx - 3, 4, k)];
        q[IDX3(q, nx, ny, nk, nx - 2, 1, k)] = q[IDX3(q, nx, ny, nk, nx - 4, 4, k)];
        q[IDX3(q, nx, ny, nk, nx - 4, 2, k)] = q[IDX3(q, nx, ny, nk, nx - 2, 5, k)];
        q[IDX3(q, nx, ny, nk, nx - 3, 2, k)] = q[IDX3(q, nx, ny, nk, nx - 3, 5, k)];
        q[IDX3(q, nx, ny, nk, nx - 2, 2, k)] = q[IDX3(q, nx, ny, nk, nx - 4, 5, k)];

        q[IDX3(q, nx, ny, nk, nx - 4, ny - 4, k)] = q[IDX3(q, nx, ny, nk, nx - 2, ny - 5, k)];
        q[IDX3(q, nx, ny, nk, nx - 4, ny - 3, k)] = q[IDX3(q, nx, ny, nk, nx - 3, ny - 5, k)];
        q[IDX3(q, nx, ny, nk, nx - 4, ny - 2, k)] = q[IDX3(q, nx, ny, nk, nx - 4, ny - 5, k)];
        q[IDX3(q, nx, ny, nk, nx - 3, ny - 4, k)] = q[IDX3(q, nx, ny, nk, nx - 2, ny - 6, k)];
        q[IDX3(q, nx, ny, nk, nx - 3, ny - 3, k)] = q[IDX3(q, nx, ny, nk, nx - 3, ny - 6, k)];
        q[IDX3(q, nx, ny, nk, nx - 3, ny - 2, k)] = q[IDX3(q, nx, ny, nk, nx - 4, ny - 6, k)];
        q[IDX3(q, nx, ny, nk, nx - 2, ny - 4, k)] = q[IDX3(q, nx, ny, nk, nx - 2, ny - 7, k)];
        q[IDX3(q, nx, ny, nk, nx - 2, ny - 3, k)] = q[IDX3(q, nx, ny, nk, nx - 3, ny - 7, k)];
        q[IDX3(q, nx, ny, nk, nx - 2, ny - 2, k)] = q[IDX3(q, nx, ny, nk, nx - 4, ny - 7, k)];
    }
}

/* ------------------------------------------------------------------ */
/* compute_al_x: interpolate q to x-interfaces into al.               */
static void compute_al_x(const double *restrict q, const double *restrict dxa,
                         double *restrict al,
                         int64_t nx, int64_t ny, int64_t nk,
                         int64_t i_start, int64_t i_end, int grid_type)
{
    int64_t lo = i_start - 1;
    int64_t hi = i_end + 3;

    #pragma omp parallel for
    for (int64_t i = lo; i < hi; ++i) {
        for (int64_t j = 0; j < ny; ++j) {
            #pragma omp simd
            for (int64_t k = 0; k < nk; ++k) {
                al[IDX3(al, nx, ny, nk, i, j, k)] =
                    P1 * (q[IDX3(q, nx, ny, nk, i - 1, j, k)] + q[IDX3(q, nx, ny, nk, i, j, k)])
                  + P2 * (q[IDX3(q, nx, ny, nk, i - 2, j, k)] + q[IDX3(q, nx, ny, nk, i + 1, j, k)]);
            }
        }
    }

    if (grid_type < 3) {
        /* ia = [i_start-1, i_end] */
        #pragma omp parallel for
        for (int64_t j = 0; j < ny; ++j) {
            #pragma omp simd
            for (int64_t k = 0; k < nk; ++k) {
                int64_t i;
                i = i_start - 1;
                al[IDX3(al, nx, ny, nk, i, j, k)] =
                    C1 * q[IDX3(q, nx, ny, nk, i - 2, j, k)]
                  + C2 * q[IDX3(q, nx, ny, nk, i - 1, j, k)]
                  + C3 * q[IDX3(q, nx, ny, nk, i, j, k)];
                i = i_end;
                al[IDX3(al, nx, ny, nk, i, j, k)] =
                    C1 * q[IDX3(q, nx, ny, nk, i - 2, j, k)]
                  + C2 * q[IDX3(q, nx, ny, nk, i - 1, j, k)]
                  + C3 * q[IDX3(q, nx, ny, nk, i, j, k)];
            }
        }

        /* ib = [i_start, i_end+1]: weighted average of one-sided interpolants */
        #pragma omp parallel for
        for (int64_t j = 0; j < ny; ++j) {
            #pragma omp simd
            for (int64_t k = 0; k < nk; ++k) {
                int64_t i;
                double left, right, t1, t2, n1, n2;

                i = i_start;
                t1 = dxa[IDX3(dxa, nx, ny, nk, i - 2, j, k)] + dxa[IDX3(dxa, nx, ny, nk, i - 1, j, k)];
                t2 = dxa[IDX3(dxa, nx, ny, nk, i, j, k)] + dxa[IDX3(dxa, nx, ny, nk, i + 1, j, k)];
                n1 = (t1 + dxa[IDX3(dxa, nx, ny, nk, i - 1, j, k)]) * q[IDX3(q, nx, ny, nk, i - 1, j, k)]
                   - dxa[IDX3(dxa, nx, ny, nk, i - 1, j, k)] * q[IDX3(q, nx, ny, nk, i - 2, j, k)];
                n2 = (t2 + dxa[IDX3(dxa, nx, ny, nk, i, j, k)]) * q[IDX3(q, nx, ny, nk, i, j, k)]
                   - dxa[IDX3(dxa, nx, ny, nk, i, j, k)] * q[IDX3(q, nx, ny, nk, i + 1, j, k)];
                left = n1 / t1;
                right = n2 / t2;
                al[IDX3(al, nx, ny, nk, i, j, k)] = 0.5 * (left + right);

                i = i_end + 1;
                t1 = dxa[IDX3(dxa, nx, ny, nk, i - 2, j, k)] + dxa[IDX3(dxa, nx, ny, nk, i - 1, j, k)];
                t2 = dxa[IDX3(dxa, nx, ny, nk, i, j, k)] + dxa[IDX3(dxa, nx, ny, nk, i + 1, j, k)];
                n1 = (t1 + dxa[IDX3(dxa, nx, ny, nk, i - 1, j, k)]) * q[IDX3(q, nx, ny, nk, i - 1, j, k)]
                   - dxa[IDX3(dxa, nx, ny, nk, i - 1, j, k)] * q[IDX3(q, nx, ny, nk, i - 2, j, k)];
                n2 = (t2 + dxa[IDX3(dxa, nx, ny, nk, i, j, k)]) * q[IDX3(q, nx, ny, nk, i, j, k)]
                   - dxa[IDX3(dxa, nx, ny, nk, i, j, k)] * q[IDX3(q, nx, ny, nk, i + 1, j, k)];
                left = n1 / t1;
                right = n2 / t2;
                al[IDX3(al, nx, ny, nk, i, j, k)] = 0.5 * (left + right);
            }
        }

        /* ic = [i_start+1, i_end+2] */
        #pragma omp parallel for
        for (int64_t j = 0; j < ny; ++j) {
            #pragma omp simd
            for (int64_t k = 0; k < nk; ++k) {
                int64_t i;
                i = i_start + 1;
                al[IDX3(al, nx, ny, nk, i, j, k)] =
                    C3 * q[IDX3(q, nx, ny, nk, i - 1, j, k)]
                  + C2 * q[IDX3(q, nx, ny, nk, i, j, k)]
                  + C1 * q[IDX3(q, nx, ny, nk, i + 1, j, k)];
                i = i_end + 2;
                al[IDX3(al, nx, ny, nk, i, j, k)] =
                    C3 * q[IDX3(q, nx, ny, nk, i - 1, j, k)]
                  + C2 * q[IDX3(q, nx, ny, nk, i, j, k)]
                  + C1 * q[IDX3(q, nx, ny, nk, i + 1, j, k)];
            }
        }
    }
}

/* compute_al_y: mirror of compute_al_x in j. */
static void compute_al_y(const double *restrict q, const double *restrict dya,
                         double *restrict al,
                         int64_t nx, int64_t ny, int64_t nk,
                         int64_t j_start, int64_t j_end, int grid_type)
{
    int64_t lo = j_start - 1;
    int64_t hi = j_end + 3;

    #pragma omp parallel for
    for (int64_t i = 0; i < nx; ++i) {
        for (int64_t j = lo; j < hi; ++j) {
            #pragma omp simd
            for (int64_t k = 0; k < nk; ++k) {
                al[IDX3(al, nx, ny, nk, i, j, k)] =
                    P1 * (q[IDX3(q, nx, ny, nk, i, j - 1, k)] + q[IDX3(q, nx, ny, nk, i, j, k)])
                  + P2 * (q[IDX3(q, nx, ny, nk, i, j - 2, k)] + q[IDX3(q, nx, ny, nk, i, j + 1, k)]);
            }
        }
    }

    if (grid_type < 3) {
        /* ja = [j_start-1, j_end] */
        #pragma omp parallel for
        for (int64_t i = 0; i < nx; ++i) {
            #pragma omp simd
            for (int64_t k = 0; k < nk; ++k) {
                int64_t j;
                j = j_start - 1;
                al[IDX3(al, nx, ny, nk, i, j, k)] =
                    C1 * q[IDX3(q, nx, ny, nk, i, j - 2, k)]
                  + C2 * q[IDX3(q, nx, ny, nk, i, j - 1, k)]
                  + C3 * q[IDX3(q, nx, ny, nk, i, j, k)];
                j = j_end;
                al[IDX3(al, nx, ny, nk, i, j, k)] =
                    C1 * q[IDX3(q, nx, ny, nk, i, j - 2, k)]
                  + C2 * q[IDX3(q, nx, ny, nk, i, j - 1, k)]
                  + C3 * q[IDX3(q, nx, ny, nk, i, j, k)];
            }
        }

        /* jb = [j_start, j_end+1] */
        #pragma omp parallel for
        for (int64_t i = 0; i < nx; ++i) {
            #pragma omp simd
            for (int64_t k = 0; k < nk; ++k) {
                int64_t j;
                double left, right, t1, t2, n1, n2;

                j = j_start;
                t1 = dya[IDX3(dya, nx, ny, nk, i, j - 2, k)] + dya[IDX3(dya, nx, ny, nk, i, j - 1, k)];
                t2 = dya[IDX3(dya, nx, ny, nk, i, j, k)] + dya[IDX3(dya, nx, ny, nk, i, j + 1, k)];
                n1 = (t1 + dya[IDX3(dya, nx, ny, nk, i, j - 1, k)]) * q[IDX3(q, nx, ny, nk, i, j - 1, k)]
                   - dya[IDX3(dya, nx, ny, nk, i, j - 1, k)] * q[IDX3(q, nx, ny, nk, i, j - 2, k)];
                n2 = (t2 + dya[IDX3(dya, nx, ny, nk, i, j, k)]) * q[IDX3(q, nx, ny, nk, i, j, k)]
                   - dya[IDX3(dya, nx, ny, nk, i, j, k)] * q[IDX3(q, nx, ny, nk, i, j + 1, k)];
                left = n1 / t1;
                right = n2 / t2;
                al[IDX3(al, nx, ny, nk, i, j, k)] = 0.5 * (left + right);

                j = j_end + 1;
                t1 = dya[IDX3(dya, nx, ny, nk, i, j - 2, k)] + dya[IDX3(dya, nx, ny, nk, i, j - 1, k)];
                t2 = dya[IDX3(dya, nx, ny, nk, i, j, k)] + dya[IDX3(dya, nx, ny, nk, i, j + 1, k)];
                n1 = (t1 + dya[IDX3(dya, nx, ny, nk, i, j - 1, k)]) * q[IDX3(q, nx, ny, nk, i, j - 1, k)]
                   - dya[IDX3(dya, nx, ny, nk, i, j - 1, k)] * q[IDX3(q, nx, ny, nk, i, j - 2, k)];
                n2 = (t2 + dya[IDX3(dya, nx, ny, nk, i, j, k)]) * q[IDX3(q, nx, ny, nk, i, j, k)]
                   - dya[IDX3(dya, nx, ny, nk, i, j, k)] * q[IDX3(q, nx, ny, nk, i, j + 1, k)];
                left = n1 / t1;
                right = n2 / t2;
                al[IDX3(al, nx, ny, nk, i, j, k)] = 0.5 * (left + right);
            }
        }

        /* jc = [j_start+1, j_end+2] */
        #pragma omp parallel for
        for (int64_t i = 0; i < nx; ++i) {
            #pragma omp simd
            for (int64_t k = 0; k < nk; ++k) {
                int64_t j;
                j = j_start + 1;
                al[IDX3(al, nx, ny, nk, i, j, k)] =
                    C3 * q[IDX3(q, nx, ny, nk, i, j - 1, k)]
                  + C2 * q[IDX3(q, nx, ny, nk, i, j, k)]
                  + C1 * q[IDX3(q, nx, ny, nk, i, j + 1, k)];
                j = j_end + 2;
                al[IDX3(al, nx, ny, nk, i, j, k)] =
                    C3 * q[IDX3(q, nx, ny, nk, i, j - 1, k)]
                  + C2 * q[IDX3(q, nx, ny, nk, i, j, k)]
                  + C1 * q[IDX3(q, nx, ny, nk, i, j + 1, k)];
            }
        }
    }
}

