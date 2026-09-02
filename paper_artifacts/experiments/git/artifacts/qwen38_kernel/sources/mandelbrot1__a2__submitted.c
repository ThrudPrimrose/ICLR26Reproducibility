#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* Prevent FMA contraction of the linspace "i*step + start" so it keeps numpy's
 * two separate roundings (vectorized mul ufunc, then += start ufunc). */
static __attribute__((noinline)) double __noadd(double a, double b){ return a + b; }

/* Canonical C-ABI entry point (binding derived from the manifest):
 *   pointers (name-sorted): N_out (int64), Z_out (complex128)
 *   scalars  (name-sorted): maxiter, xn, yn  (all int64)
 *   trailing reserved pair: workspace, workspace_size
 * The viewport / horizon are pinned constants in the manifest. */
void mandelbrot1_fp64(int64_t *restrict N_out, double _Complex *restrict Z_out,
                      const int64_t maxiter, const int64_t xn, const int64_t yn,
                      uint8_t *restrict workspace, const int64_t workspace_size){
    (void)workspace; (void)workspace_size;
    const double xmin=-2.25, xmax=0.75, ymin=-1.25, ymax=1.25, horizon2=4.0;

    double *X = (double*)malloc((size_t)xn * sizeof(double));
    double *Y = (double*)malloc((size_t)yn * sizeof(double));
    double dx = xmax - xmin, sx = dx / (double)(xn - 1);
    for (int64_t j=0; j<xn-1; j++) X[j] = __noadd((double)j * sx, xmin);
    X[xn-1] = xmax;
    double dy = ymax - ymin, sy = dy / (double)(yn - 1);
    for (int64_t i=0; i<yn-1; i++) Y[i] = __noadd((double)i * sy, ymin);
    Y[yn-1] = ymax;

    #pragma omp parallel for schedule(dynamic, 4)
    for (int64_t i=0; i<yn; i++){
        const double Ci = Y[i];
        int64_t *Nrow = N_out + i*xn;
        double _Complex *Zrow = Z_out + i*xn;
        for (int64_t j=0; j<xn; j++){
            const double Cr = X[j];
            double zr = 0.0, zi = 0.0;
            int64_t n = 0;
            while (n < maxiter){
                const double mag2 = zr*zr + zi*zi;
                if (!(mag2 < horizon2)) break;   /* matches numpy: stay iff mag2<horizon2 */
                Nrow[j] = n;
                /* complex square matching numpy's vectorized FMA: re=fma(a,a,-b*b), im=fma(a,b,a*b) */
                const double nre = fma(zr, zr, -(zi*zi));
                const double nim = fma(zr, zi, (zr*zi));
                zr = nre + Cr;
                zi = nim + Ci;
                n++;
            }
            if (Nrow[j] == maxiter - 1) Nrow[j] = 0;
            union { double _Complex c; double p[2]; } u;
            u.p[0] = zr; u.p[1] = zi;
            Zrow[j] = u.c;
        }
    }
    free(X); free(Y);
}
