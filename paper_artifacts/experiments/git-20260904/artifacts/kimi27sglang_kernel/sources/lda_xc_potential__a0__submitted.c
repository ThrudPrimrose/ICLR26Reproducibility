#include <stdint.h>
#include <math.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void lda_xc_potential_fp64(double *restrict exc,
                           const double *restrict rho,
                           double *restrict vxc,
                           const int64_t N,
                           const double dvol)
{
    const double AX = 0.9847450218426965;
    const double A  = 0.0311;
    const double B  = -0.0480;
    const double Cc = 0.0020;
    const double D  = -0.0116;
    const double C3 = 3.0 / (4.0 * M_PI);
    const double nmin = 1.0e-12;

    const int64_t NN = N * N;
    double exc_sum = 0.0;

    #pragma omp parallel for schedule(static) reduction(+:exc_sum)
    for (int64_t i = 0; i < N; ++i) {
        const int64_t base_i = i * NN;
        for (int64_t j = 0; j < N; ++j) {
            const int64_t base = base_i + j * N;
            #pragma omp simd reduction(+:exc_sum)
            for (int64_t k = 0; k < N; ++k) {
                double n = rho[base + k];
                if (n < nmin) n = nmin;

                double n13 = cbrt(n);
                double eps_x = -0.75 * AX * n13;
                double v_x   = -AX * n13;

                double rs = cbrt(C3 / n);
                double ln_rs = log(rs);

                double eps_c = A * ln_rs + B + Cc * rs * ln_rs + D * rs;
                double v_c   = A * ln_rs + (B - A / 3.0)
                               + (2.0 / 3.0) * Cc * rs * ln_rs
                               + ((2.0 * D - Cc) / 3.0) * rs;

                vxc[base + k] = v_x + v_c;
                exc_sum += (eps_x + eps_c) * n;
            }
        }
    }

    exc[0] = dvol * exc_sum;
}
