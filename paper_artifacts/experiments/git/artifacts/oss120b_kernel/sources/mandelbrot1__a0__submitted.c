#define _USE_MATH_DEFINES
#include <stdint.h>
#include <complex.h>

/* Mandelbrot set kernel (double precision).
   Implements the reference algorithm from mandelbrot1_numpy.py.
   Parameters:
     xmin, xmax, ymin, ymax : bounds of the sampling region.
     xn, yn                 : number of samples along the X and Y axes.
     maxiter                : maximum iteration count.
     horizon                : escape radius.
     Z_out                  : output array of complex values, size yn*xn, row-major.
     N_out                  : output array of iteration counts, same size.
   The implementation uses explicit real/imag arithmetic and an early‑exit
   optimisation that writes the iteration count only once per pixel.
   OpenMP parallelises over rows for good scaling on many cores.
*/

void mandelbrot1_fp64(double xmin, double xmax, double ymin, double ymax,
                      int64_t xn, int64_t yn, int64_t maxiter, double horizon,
                      double _Complex *restrict Z_out,
                      int64_t *restrict N_out) {
    const double dx = (xn > 1) ? (xmax - xmin) / (xn - 1) : 0.0;
    const double dy = (yn > 1) ? (ymax - ymin) / (yn - 1) : 0.0;
    const double horizon2 = horizon * horizon;

    #pragma omp parallel for schedule(static)
    for (int64_t iy = 0; iy < yn; ++iy) {
        double y = ymin + iy * dy;
        int64_t row_offset = iy * xn;
        for (int64_t ix = 0; ix < xn; ++ix) {
            double x = xmin + ix * dx;
            double Zr = 0.0, Zi = 0.0;
            int64_t n;
            int64_t last_n = 0;
            for (n = 0; n < maxiter; ++n) {
                double mag2 = Zr * Zr + Zi * Zi;
                if (mag2 >= horizon2) {
                    break;
                }
                last_n = n;
                double Zr_new = Zr * Zr - Zi * Zi + x;
                double Zi_new = 2.0 * Zr * Zi + y;
                Zr = Zr_new;
                Zi = Zi_new;
            }
            int64_t idx = row_offset + ix;
            if (n == maxiter) {
                N_out[idx] = 0; /* never escaped */
            } else {
                N_out[idx] = last_n;
            }
            Z_out[idx] = Zr + Zi * I;
        }
    }
}
