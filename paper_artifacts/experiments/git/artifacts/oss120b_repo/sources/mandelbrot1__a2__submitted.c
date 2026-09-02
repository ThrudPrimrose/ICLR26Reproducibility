// Optimized mandelbrot1 implementation (hand-tuned)
#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

/* ``z.conjugate()`` -- named helper so the C and C++ preludes
 * offer the same spelling. C has the standard one: ``conj``
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
#ifndef min
#define min(a, b) ((((a) != (a)) || ((b) != (b))) ? ((a) + (b)) : (((b) < (a)) ? (b) : (a)))
#endif
#ifndef max
#define max(a, b) ((((a) != (a)) || ((b) != (b))) ? ((a) + (b)) : (((b) > (a)) ? (b) : (a)))
#endif
static inline double __npb_fmax_f(double a, double b) { return (a != a) ? a : (b != b) ? b : (a > b ? a : b); }
static inline double __npb_fmin_f(double a, double b) { return (a != a) ? a : (b != b) ? b : (a < b ? a : b); }
static inline int64_t __npb_fmax_i(int64_t a, int64_t b) { return a > b ? a : b; }
static inline int64_t __npb_fmin_i(int64_t a, int64_t b) { return a < b ? a : b; }
static inline uint64_t __npb_fmax_u(uint64_t a, uint64_t b) { return a > b ? a : b; }
static inline uint64_t __npb_fmin_u(uint64_t a, uint64_t b) { return a < b ? a : b; }
static inline double __npb_sign(double x) { return x != x ? x : (double)((x > 0) - (x < 0)); }
static inline int64_t __npb_floordiv_i(int64_t a, int64_t b) { return a / b - ((a % b != 0) && ((a < 0) ^ (b < 0))); }
static inline double __npb_floordiv_f(double a, double b) { return floor(a / b); }
static inline uint64_t __npb_floordiv_u(uint64_t a, uint64_t b) { return a / b; }
static inline uint64_t __npb_ceildiv_u(uint64_t a, uint64_t b) { return a / b + (a % b != 0); }
static inline uint64_t __npb_mod_u(uint64_t a, uint64_t b) { return a % b; }
#if defined(__FLT16_MANT_DIG__)
#define __NPB_F16_ASSOC(fn) _Float16: fn,
#else
#define __NPB_F16_ASSOC(fn)
#endif
#define __NPB_UNSIGNED_ASSOC(fn) unsigned int: fn, unsigned long: fn, unsigned long long: fn,
#define __npb_fmin(a, b) _Generic((a) + (b), \
    __NPB_F16_ASSOC(__npb_fmin_f) \
    __NPB_UNSIGNED_ASSOC(__npb_fmin_u) \
    float: __npb_fmin_f, double: __npb_fmin_f, long double: __npb_fmin_f, \
    default: __npb_fmin_i)((a), (b))
#define __npb_fmax(a, b) _Generic((a) + (b), \
    __NPB_F16_ASSOC(__npb_fmax_f) \
    __NPB_UNSIGNED_ASSOC(__npb_fmax_u) \
    float: __npb_fmax_f, double: __npb_fmax_f, long double: __npb_fmax_f, \
    default: __npb_fmax_i)((a), (b))
#ifndef int_floor
#define int_floor(a, b) _Generic((a) + (b), \
    __NPB_F16_ASSOC(__npb_floordiv_f) \
    __NPB_UNSIGNED_ASSOC(__npb_floordiv_u) \
    float: __npb_floordiv_f, double: __npb_floordiv_f, long double: __npb_floordiv_f, \
    default: __npb_floordiv_i)((a), (b))
#endif
static inline int64_t __npb_ceildiv_i(int64_t a, int64_t b) { return a / b + ((a % b != 0) && ((a < 0) == (b < 0))); }
static inline double __npb_ceildiv_f(double a, double b) { return ceil(a / b); }
#ifndef int_ceil
#define int_ceil(a, b) _Generic((a) + (b), \
    __NPB_F16_ASSOC(__npb_ceildiv_f) \
    __NPB_UNSIGNED_ASSOC(__npb_ceildiv_u) \
    float: __npb_ceildiv_f, double: __npb_ceildiv_f, long double: __npb_ceildiv_f, \
    default: __npb_ceildiv_i)((a), (b))
#endif
#ifndef floord
static inline int64_t floord(int64_t a, int64_t b) { return __npb_floordiv_i(a, b); }
#endif
#ifndef ceild
static inline int64_t ceild(int64_t a, int64_t b) { return __npb_ceildiv_i(a, b); }
#endif
static inline int64_t __npb_mod_i(int64_t a, int64_t b) { return (a % b + b) % b; }
static inline double python_fmod(double a, double b) { double m = fmod(a, b); if (m != 0.0 && ((b < 0.0) != (m < 0.0))) m += b; return m; }
#ifndef python_mod
#define python_mod(a, b) _Generic((a) + (b), \
    __NPB_F16_ASSOC(python_fmod) \
    __NPB_UNSIGNED_ASSOC(__npb_mod_u) \
    float: python_fmod, double: python_fmod, long double: python_fmod, \
    default: __npb_mod_i)((a), (b))
#endif
static inline int64_t __npb_int_pow(int64_t base, int64_t exp) { int64_t result = 1; while (exp > 0) { if (exp & 1) result *= base; base *= base; exp >>= 1; } return result; }

constexpr double horizon = 2.0;
constexpr double xmax = 0.75;
constexpr double xmin = -2.25;
constexpr double ymax = 1.25;
constexpr double ymin = -1.25;

void mandelbrot1_fp64(int64_t *restrict N_out, double _Complex *restrict Z_out, const int64_t maxiter, const int64_t xn, const int64_t yn) {
    const double horizon2 = horizon * horizon;
    const int64_t nxy = xn * yn;
    // Allocate arrays
    int64_t *N = (int64_t *)malloc((size_t)nxy * sizeof(int64_t));
    double _Complex *Z = (double _Complex *)malloc((size_t)nxy * sizeof(double _Complex));
    double *X = (double *)malloc((size_t)xn * sizeof(double));
    double *Y = (double *)malloc((size_t)yn * sizeof(double));
    double _Complex *C = (double _Complex *)malloc((size_t)nxy * sizeof(double _Complex));
    if (!N || !Z || !X || !Y || !C) {
        // Allocation failure – abort (undefined behavior in reference, but keep it simple)
        abort();
    }
    // Initialize N and Z to zero
    memset(N, 0, (size_t)nxy * sizeof(int64_t));
    memset(Z, 0, (size_t)nxy * sizeof(double _Complex));
    // Compute linearly spaced coordinates
    double stepX = (xmax - xmin) / ((xn > 1) ? (xn - 1) : 1);
    for (int64_t i = 0; i < xn; ++i) {
        X[i] = xmin + i * stepX;
    }
    if (xn > 1) X[xn - 1] = xmax;
    double stepY = (ymax - ymin) / ((yn > 1) ? (yn - 1) : 1);
    for (int64_t i = 0; i < yn; ++i) {
        Y[i] = ymin + i * stepY;
    }
    if (yn > 1) Y[yn - 1] = ymax;
    // Compute complex constant grid C = X + Y*i
    #pragma omp parallel for schedule(static)
    for (int64_t iy = 0; iy < yn; ++iy) {
        double y = Y[iy];
        for (int64_t ix = 0; ix < xn; ++ix) {
            C[iy * xn + ix] = X[ix] + y * _Complex_I;
        }
    }
    // Main iteration loop
    for (int64_t n = 0; n < maxiter; ++n) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < nxy; ++i) {
            double zr = creal(Z[i]);
            double zi = cimag(Z[i]);
            double mag2 = zr * zr + zi * zi;
            if (mag2 < horizon2) {
                N[i] = n;
                Z[i] = Z[i] * Z[i] + C[i];
            }
        }
    }
    // Points that never escaped get N = 0 (reference logic)
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < nxy; ++i) {
        if (N[i] == maxiter - 1) {
            N[i] = 0;
        }
    }
    // Copy results to output buffers
    memcpy(Z_out, Z, (size_t)nxy * sizeof(double _Complex));
    memcpy(N_out, N, (size_t)nxy * sizeof(int64_t));
    // Free temporary buffers
    free(N);
    free(Z);
    free(X);
    free(Y);
    free(C);
}

