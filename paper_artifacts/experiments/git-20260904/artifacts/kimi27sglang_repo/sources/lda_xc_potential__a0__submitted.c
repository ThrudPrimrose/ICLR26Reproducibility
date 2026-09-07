// hpcagent_bench-autogen -- generated from lda_xc_potential_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

/* ``z.conjugate()`` -- named helper so the C and C++ preludes
 * offer the same spelling. C has the standard one: ``conj`
 * from <complex.h>. The C++ prelude, which has no <complex.h>,
 * writes its own. */
static inline double _Complex __npb_conj(double _Complex z) {
    return conj(z);
}
/* M_PI / M_E etc. are POSIX/GNU extensions -- ensure they
 * are defined even on strict-C builds (glibc 2.27+ /
 * BSDs / MSVC). */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif
/* ``<complex.h>`` defines ``I`` as the imaginary unit;
 * undef it so user variable names like ``I`` (mandelbrot
 * boolean mask) don''t collide. Complex literals continue
 * to use the portable ``_Complex_I`` form. */
#ifdef I
#undef I
#endif
/* ``max``/``min`` PROPAGATE NaN (a NaN in EITHER operand yields NaN):
 * these serve the elementwise ``np.maximum``/``np.minimum`` broadcast
 * and the ``np.maximum.at`` / ``np.minimum.at`` scatter folds, which
 * follow numpy (propagate), not Python's builtin max (which drops a NaN
 * second operand). ``(a)+(b)`` is NaN whenever either operand is; for
 * finite operands the ternary picks the larger/smaller -- identical to
 * a plain comparison, so the 3-way builtin max (needleman_wunsch, always
 * finite) is unchanged. For integer operands the NaN test is dead. */
#ifndef min
#define min(a, b) ((((a) != (a)) || ((b) != (b))) ? ((a) + (b)) : (((b) < (a)) ? (b) : (a)))
#endif
#ifndef max
#define max(a, b) ((((a) != (a)) || ((b) != (b))) ? ((a) + (b)) : (((b) > (a)) ? (b) : (a)))
#endif

void lda_xc_potential_fp64(double *restrict exc, const double *restrict rho, double *restrict vxc, const int64_t N, const double dvol) {
    const int64_t total = N * N * N;
    if (total <= 0) {
        exc[0] = 0.0;
        return;
    }

    const double AX = 0.9847450218426965;
    const double GAMMA = -0.1423;
    const double B1 = 1.0529;
    const double B2 = 0.3334;
    const double A = 0.0311;
    const double B = -0.0480;
    const double C = 0.0020;
    const double D = -0.0116;

    const double coeff = 3.0 / (4.0 * M_PI);
    const double cbrt_coeff = cbrt(coeff);
    const double c1_ge1 = (7.0 / 6.0) * B1;
    const double c2_ge1 = (4.0 / 3.0) * B2;
    const double BmA3 = B - A / 3.0;
    const double two3C = (2.0 / 3.0) * C;
    const double twoDmC_3 = (2.0 * D - C) / 3.0;

    const int nt = omp_get_max_threads();
    double *part = (double *)calloc((size_t)nt, sizeof(double));

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        double mysum = 0.0;
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < total; ++i) {
            double n = rho[i];
            if (n < 1.0e-12) n = 1.0e-12;

            double n13 = cbrt(n);
            double rs = cbrt_coeff / n13;
            double sqrt_rs = sqrt(rs);
            double ln_rs = log(rs);

            double denom = 1.0 + B1 * sqrt_rs + B2 * rs;
            double inv_denom = 1.0 / denom;

            double eps_c_ge1 = GAMMA * inv_denom;
            double v_c_ge1 = eps_c_ge1 * (1.0 + c1_ge1 * sqrt_rs + c2_ge1 * rs) * inv_denom;

            double eps_c_lt1 = A * ln_rs + B + C * rs * ln_rs + D * rs;
            double v_c_lt1 = A * ln_rs + BmA3 + two3C * rs * ln_rs + twoDmC_3 * rs;

            const int high_density = rs < 1.0;
            double eps_c = high_density ? eps_c_lt1 : eps_c_ge1;
            double v_c = high_density ? v_c_lt1 : v_c_ge1;

            double v_x = -AX * n13;
            vxc[i] = v_x + v_c;

            double eps_x = -0.75 * AX * n13;
            mysum += (eps_x + eps_c) * n;
        }
        part[tid] = mysum;
    }

    double sum = 0.0;
    for (int i = 0; i < nt; ++i) sum += part[i];
    exc[0] = dvol * sum;
    free(part);
}
