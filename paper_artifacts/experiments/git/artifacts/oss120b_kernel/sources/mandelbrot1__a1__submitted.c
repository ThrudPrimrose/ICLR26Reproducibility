#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <complex.h>

/*
 * Mandelbrot set kernel (double precision complex).
 *
 * Parameters:
 *   xmin, xmax, ymin, ymax : bounds of the complex plane region.
 *   xn, yn                 : number of samples along the x and y axes.
 *   maxiter                : maximum number of escape iterations.
 *   horizon                : escape radius (the absolute value limit).
 *   Z_out                  : output complex array of shape (yn, xn).
 *   N_out                  : output iteration count array of shape (yn, xn).
 *
 * The algorithm follows the NumPy reference implementation:
 *   – generate linearly spaced X and Y vectors,
 *   – build the complex grid C = X + Y*i,
 *   – iterate Z = Z*Z + C while |Z| < horizon, storing the iteration number.
 *   – points that never escape are marked with N = 0.
 */

void mandelbrot1_fp64(double _Complex *restrict Z_out,
                      int64_t *restrict N_out,
                      double xmin, double xmax,
                      double ymin, double ymax,
                      int64_t xn, int64_t yn,
                      int64_t maxiter,
                      double horizon)
{
    /* Compute step sizes for linspace (inclusive end points). */
    double dx = (xn > 1) ? (xmax - xmin) / (double)(xn - 1) : 0.0;
    double dy = (yn > 1) ? (ymax - ymin) / (double)(yn - 1) : 0.0;
    double horizon2 = horizon * horizon;

    /* Parallel compute per point. */
    #pragma omp parallel for schedule(static) collapse(2)
    for (int64_t i = 0; i < yn; ++i) {
        double yi = ymin + i * dy;
        for (int64_t j = 0; j < xn; ++j) {
            int64_t idx = i * xn + j;
            double xr = xmin + j * dx;
            double zr = 0.0, zi = 0.0;
            int64_t n;
            for (n = 0; n < maxiter; ++n) {
                double mag2 = zr*zr + zi*zi;
                if (mag2 >= horizon2) break;
                double zr_new = zr*zr - zi*zi + xr;
                double zi_new = 2.0*zr*zi + yi;
                zr = zr_new;
                zi = zi_new;
            }
            Z_out[idx] = zr + zi * I;
            N_out[idx] = (n == maxiter) ? 0 : n - 1;
        }
    }
}

