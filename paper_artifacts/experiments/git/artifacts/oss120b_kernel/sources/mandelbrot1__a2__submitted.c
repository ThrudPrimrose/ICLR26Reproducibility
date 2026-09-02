/* Optimized mandelbrot1 kernel implementation (double precision) */
#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <complex.h>
#include <omp.h>

/*
 * Compute the Mandelbrot set for a rectangular region.
 *
 * Arguments:
 *   Z_out    – output complex array of size [yn][xn] (row-major).
 *   N_out    – output iteration count array of size [yn][xn] (int64).
 *   xmin, xmax, ymin, ymax – real bounds of the region.
 *   xn, yn   – number of points in X and Y direction.
 *   maxiter  – maximum number of iterations.
 *   horizon  – escape radius (typically 2.0).
 *
 * The algorithm follows the reference Python implementation:
 *   for n in range(maxiter):
 *       I = |Z|^2 < horizon^2
 *       N[I] = n
 *       Z[I] = Z[I]^2 + C
 *   N[N == maxiter-1] = 0
 *
 * This C version uses explicit loops, OpenMP parallelism, and avoids
 * temporary complex allocations inside the innermost loop.
 */

void mandelbrot1_fp64(double xmin, double xmax, double ymin, double ymax,
                      int64_t xn, int64_t yn, int64_t maxiter, double horizon,
                      double _Complex *restrict Z_out, int64_t *restrict N_out) {
    if (xn <= 0 || yn <= 0 || maxiter <= 0) return;

    // Allocate and fill linearly spaced X and Y coordinate vectors.
    double *X = (double *)malloc((size_t)xn * sizeof(double));
    double *Y = (double *)malloc((size_t)yn * sizeof(double));
    if (!X || !Y) {
        free(X);
        free(Y);
        return; // allocation failure – nothing to compute.
    }

    double x_step = (xn > 1) ? (xmax - xmin) / (xn - 1) : 0.0;
    double y_step = (yn > 1) ? (ymax - ymin) / (yn - 1) : 0.0;
    for (int64_t j = 0; j < xn; ++j) {
        X[j] = xmin + (double)j * x_step;
    }
    for (int64_t i = 0; i < yn; ++i) {
        Y[i] = ymin + (double)i * y_step;
    }

    const double horizon2 = horizon * horizon;
    const int64_t total = xn * yn;

    // Initialise Z_out to zero (the reference does it via zeros).
    #pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < total; ++idx) {
        Z_out[idx] = 0.0 + 0.0 * _Complex_I;
    }

    // Main iteration loop.
    for (int64_t n = 0; n < maxiter; ++n) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < yn; ++i) {
            double Yi = Y[i];
            int64_t row_off = i * xn;
            for (int64_t j = 0; j < xn; ++j) {
                int64_t idx = row_off + j;
                double _Complex z = Z_out[idx];
                double zr = __real__ z;
                double zi = __imag__ z;
                double mag2 = zr * zr + zi * zi;
                if (mag2 < horizon2) {
                    N_out[idx] = n;
                    double Cre = X[j];
                    double Cim = Yi;
                    double new_real = zr * zr - zi * zi + Cre;
                    double new_imag = 2.0 * zr * zi + Cim;
                    Z_out[idx] = new_real + new_imag * _Complex_I;
                }
            }
        }
    }

    // Points that never escaped get iteration count maxiter-1; set them to 0.
    const int64_t maxval = maxiter - 1;
    #pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < total; ++idx) {
        if (N_out[idx] == maxval) {
            N_out[idx] = 0;
        }
    }

    free(X);
    free(Y);
}
