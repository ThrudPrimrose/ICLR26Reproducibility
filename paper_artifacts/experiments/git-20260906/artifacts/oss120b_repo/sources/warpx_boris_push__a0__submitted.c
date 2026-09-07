// hpcagent_bench-autogen -- generated from warpx_boris_push_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <complex.h>
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
/* Elementwise ``np.maximum``/``np.minimum`` lower to ``fmax``/``fmin``;
 * libm ``fmax``/``fmin`` SUPPRESS NaN (return the non-NaN operand) but
 * numpy PROPAGATES it. These single-evaluation helpers return NaN when
 * either operand is NaN, else the larger/smaller.
 * Integer operands take the INTEGER form, dispatched on the promoted operand
 * type exactly as int_floor is: routing them through the double helper rounds
 * every value above 2**53 to the nearest representable double, so
 * min(2**53 + 1, 2**53 + 2) returned 2**53 -- a value neither operand had. */
static inline double __npb_fmax_f(double a, double b) {
    return (a != a) ? a : (b != b) ? b : (a > b ? a : b);
}
static inline double __npb_fmin_f(double a, double b) {
    return (a != a) ? a : (b != b) ? b : (a < b ? a : b);
}
static inline int64_t __npb_fmax_i(int64_t a, int64_t b) { return a > b ? a : b; }
static inline int64_t __npb_fmin_i(int64_t a, int64_t b) { return a < b ? a : b; }
static inline uint64_t __npb_fmax_u(uint64_t a, uint64_t b) { return a > b ? a : b; }
static inline uint64_t __npb_fmin_u(uint64_t a, uint64_t b) { return a < b ? a : b; }
/* ``np.sign``: numpy ``sign(nan) == nan`` and ``sign(0) == 0``. The
 * naive ``(x>0)-(x<0)`` gives 0 for NaN and evaluates ``x`` twice. */
static inline double __npb_sign(double x) {
    return x != x ? x : (double)((x > 0) - (x < 0));
}
/* Python ``//`` floors toward -inf; C ``/`` truncates toward zero. Integer and
 * floating operands need different corrections, so the division helpers dispatch
 * on the PROMOTED OPERAND TYPE -- the emitter never has to infer the dtype from
 * the source AST (guessing it wrong silently truncated instead of flooring).
 * _Generic's controlling expression is unevaluated and each argument is spelled
 * once, so operands with side effects are evaluated exactly once. */
static inline int64_t __npb_floordiv_i(int64_t a, int64_t b) {
    return a / b - ((a % b != 0) && ((a < 0) ^ (b < 0)));
}
static inline double __npb_floordiv_f(double a, double b) { return floor(a / b); }
/* Unsigned operands need their own form: floor == truncate for them, and routing
 * them through the SIGNED helper reinterprets any value above INT64_MAX as
 * negative ((2**63 + 5) // 2 came back negative). */
static inline uint64_t __npb_floordiv_u(uint64_t a, uint64_t b) { return a / b; }
static inline uint64_t __npb_ceildiv_u(uint64_t a, uint64_t b) { return a / b + (a % b != 0); }
static inline uint64_t __npb_mod_u(uint64_t a, uint64_t b) { return a % b; }
/* _Float16 is NOT promoted by GCC in arithmetic, so `_Float16 + _Float16` has type
 * _Float16 and fell to `default:` -- the INTEGER helper. 0.5 // 0.25 became
 * int_floor(0, 0) and died with SIGFPE. Spelled as a macro because the association
 * only exists where the type does. */
#if defined(__FLT16_MANT_DIG__)
#define __NPB_F16_ASSOC(fn) _Float16: fn,
#else
#define __NPB_F16_ASSOC(fn)
#endif
#define __NPB_UNSIGNED_ASSOC(fn) \
    unsigned int: fn, unsigned long: fn, unsigned long long: fn,
/* min/max dispatch (declared above): integer operands stay exact, floating ones
 * propagate NaN. Spelled here because the type associations are. */
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
/* Ceil-division counterpart (toward +inf), exact for both signs -- unlike the
 * ``(a + b - 1) / b`` idiom, which is correct only for a positive divisor and
 * overflows near the integer maximum. */
static inline int64_t __npb_ceildiv_i(int64_t a, int64_t b) {
    return a / b + ((a % b != 0) && ((a < 0) == (b < 0)));
}
static inline double __npb_ceildiv_f(double a, double b) { return ceil(a / b); }
#ifndef int_ceil
#define int_ceil(a, b) _Generic((a) + (b), \
    __NPB_F16_ASSOC(__npb_ceildiv_f) \
    __NPB_UNSIGNED_ASSOC(__npb_ceildiv_u) \
    float: __npb_ceildiv_f, double: __npb_ceildiv_f, long double: __npb_ceildiv_f, \
    default: __npb_ceildiv_i)((a), (b))
#endif
/* pet's named quasi-affine builtins (POLYCC-008); guarded because polycc prepends
 * its own #define floord/ceild, which would expand these declarators (POLYCC-004). */
#ifndef floord
static inline int64_t floord(int64_t a, int64_t b) {
    return __npb_floordiv_i(a, b);
}
#endif
#ifndef ceild
static inline int64_t ceild(int64_t a, int64_t b) {
    return __npb_ceildiv_i(a, b);
}
#endif
/* Python ``%`` returns sign of divisor; C returns sign of dividend. Same
 * type-dispatch as int_floor: integer operands use the exact integer form,
 * floating operands numpy's npy_remainder (see python_fmod). */
static inline int64_t __npb_mod_i(int64_t a, int64_t b) { return (a % b + b) % b; }
/* Floating-point ``%``: numpy's floored modulo takes the sign of the
 * divisor, which integer ``python_mod`` cannot express on doubles.
 * Mirrors numpy ``npy_remainder`` (fmod + sign-of-divisor fixup). */
static inline double python_fmod(double a, double b) {
    double m = fmod(a, b);
    if (m != 0.0 && ((b < 0.0) != (m < 0.0))) m += b;
    return m;
}
#ifndef python_mod
#define python_mod(a, b) _Generic((a) + (b), \
    __NPB_F16_ASSOC(python_fmod) \
    __NPB_UNSIGNED_ASSOC(__npb_mod_u) \
    float: python_fmod, double: python_fmod, long double: python_fmod, \
    default: __npb_mod_i)((a), (b))
#endif
/* Integer power for VLA shape bounds like ``R ** K``. */
static inline int64_t __npb_int_pow(int64_t base, int64_t exp) {
    int64_t result = 1;
    while (exp > 0) {
        if (exp & 1) result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
}

constexpr double dt = 1e-13;

void warpx_boris_push_fp64_old(const double *restrict Bx, const double *restrict By, const double *restrict Bz, const double *restrict Ex, const double *restrict Ey, const double *restrict Ez, double *restrict ux, double *restrict uy, double *restrict uz, const double m, const int64_t momentum_push_type, const int64_t np_particles, const double q) {
        int64_t mpt;
        double __inl1_econst;
        double __inl1_inv_c2;
        double *__cb1 = (double *)malloc((size_t)((np_particles)) * sizeof(double));
        double *__inl1_inv_gamma = (double *)malloc((size_t)((np_particles)) * sizeof(double));
        double *__inl1_tx = (double *)malloc((size_t)((np_particles)) * sizeof(double));
        double *__inl1_ty = (double *)malloc((size_t)((np_particles)) * sizeof(double));
        double *__inl1_tz = (double *)malloc((size_t)((np_particles)) * sizeof(double));
        double *__inl1_tsqi = (double *)malloc((size_t)((np_particles)) * sizeof(double));
        double *__inl1_sx = (double *)malloc((size_t)((np_particles)) * sizeof(double));
        double *__inl1_sy = (double *)malloc((size_t)((np_particles)) * sizeof(double));
        double *__inl1_sz = (double *)malloc((size_t)((np_particles)) * sizeof(double));
        double *__inl1_ux_p = (double *)malloc((size_t)((np_particles)) * sizeof(double));
        double *__inl1_uy_p = (double *)malloc((size_t)((np_particles)) * sizeof(double));
        double *__inl1_uz_p = (double *)malloc((size_t)((np_particles)) * sizeof(double));
        mpt = ((int64_t)(momentum_push_type));
        __inl1_econst = (((0.5 * q) * dt) / m);
        if (((mpt == 1) || (mpt == 0))) {
          for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
            ux[__w0] += (__inl1_econst * Ex[__w0]);
          }
          for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
            uy[__w0] += (__inl1_econst * Ey[__w0]);
          }
          for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
            uz[__w0] += (__inl1_econst * Ez[__w0]);
          }
        }
        __inl1_inv_c2 = 1.1126500560536185e-17;
        /* numpy: np.sqrt(1.0 + (ux * ux + uy * uy + uz * uz) * __inl1_inv_c2) */
        for (int64_t __r0 = 0; __r0 < np_particles; ++__r0) {
          __cb1[__r0] = sqrt((1.0 + ((((ux[__r0] * ux[__r0]) + (uy[__r0] * uy[__r0])) + (uz[__r0] * uz[__r0])) * __inl1_inv_c2)));
        }
        for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
          __inl1_inv_gamma[__w0] = (1.0 / __cb1[__w0]);
        }
        for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
          __inl1_tx[__w0] = ((__inl1_econst * __inl1_inv_gamma[__w0]) * Bx[__w0]);
        }
        for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
          __inl1_ty[__w0] = ((__inl1_econst * __inl1_inv_gamma[__w0]) * By[__w0]);
        }
        for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
          __inl1_tz[__w0] = ((__inl1_econst * __inl1_inv_gamma[__w0]) * Bz[__w0]);
        }
        if (((mpt == 1) || (mpt == 2))) {
          double *__inl1_tsq = (double *)malloc(((np_particles)) * sizeof(double));
          for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
            __inl1_tsq[__w0] = (((__inl1_tx[__w0] * __inl1_tx[__w0]) + (__inl1_ty[__w0] * __inl1_ty[__w0])) + (__inl1_tz[__w0] * __inl1_tz[__w0]));
          }
          bool *__inl1_has_field = (bool *)malloc(((np_particles)) * sizeof(bool));
          for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
            __inl1_has_field[__w0] = (__inl1_tsq[__w0] > 0.0);
          }
          double *__cb2 = (double *)malloc(((np_particles)) * sizeof(double));
          /* numpy: np.where(__inl1_has_field, __inl1_tsq, 1.0) */
          for (int64_t __r0 = 0; __r0 < np_particles; ++__r0) {
            __cb2[__r0] = (__inl1_has_field[__r0] ? __inl1_tsq[__r0] : 1.0);
          }
          double *__inl1_safe_tsq = (double *)malloc(((np_particles)) * sizeof(double));
          for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
            __inl1_safe_tsq[__w0] = __cb2[__w0];
          }
          double *__cb3 = (double *)malloc(((np_particles)) * sizeof(double));
          /* numpy: np.where(__inl1_has_field, (sqrt(1.0 + __inl1_tsq) - 1.0) / __inl1_safe_tsq, 0.5) */
          for (int64_t __r0 = 0; __r0 < np_particles; ++__r0) {
            __cb3[__r0] = (__inl1_has_field[__r0] ? ((sqrt((1.0 + __inl1_tsq[__r0])) - 1.0) / __inl1_safe_tsq[__r0]) : 0.5);
          }
          double *__inl1_factor = (double *)malloc(((np_particles)) * sizeof(double));
          for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
            __inl1_factor[__w0] = __cb3[__w0];
          }
          for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
            __inl1_tx[__w0] *= __inl1_factor[__w0];
          }
          for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
            __inl1_ty[__w0] *= __inl1_factor[__w0];
          }
          for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
            __inl1_tz[__w0] *= __inl1_factor[__w0];
          }
          free(__inl1_tsq);
          free(__inl1_has_field);
          free(__cb2);
          free(__inl1_safe_tsq);
          free(__cb3);
          free(__inl1_factor);
        }
        for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
          __inl1_tsqi[__w0] = (2.0 / (((1.0 + (__inl1_tx[__w0] * __inl1_tx[__w0])) + (__inl1_ty[__w0] * __inl1_ty[__w0])) + (__inl1_tz[__w0] * __inl1_tz[__w0])));
        }
        for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
          __inl1_sx[__w0] = (__inl1_tx[__w0] * __inl1_tsqi[__w0]);
        }
        for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
          __inl1_sy[__w0] = (__inl1_ty[__w0] * __inl1_tsqi[__w0]);
        }
        for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
          __inl1_sz[__w0] = (__inl1_tz[__w0] * __inl1_tsqi[__w0]);
        }
        for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
          __inl1_ux_p[__w0] = ((ux[__w0] + (uy[__w0] * __inl1_tz[__w0])) - (uz[__w0] * __inl1_ty[__w0]));
        }
        for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
          __inl1_uy_p[__w0] = ((uy[__w0] + (uz[__w0] * __inl1_tx[__w0])) - (ux[__w0] * __inl1_tz[__w0]));
        }
        for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
          __inl1_uz_p[__w0] = ((uz[__w0] + (ux[__w0] * __inl1_ty[__w0])) - (uy[__w0] * __inl1_tx[__w0]));
        }
        for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
          ux[__w0] += ((__inl1_uy_p[__w0] * __inl1_sz[__w0]) - (__inl1_uz_p[__w0] * __inl1_sy[__w0]));
        }
        for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
          uy[__w0] += ((__inl1_uz_p[__w0] * __inl1_sx[__w0]) - (__inl1_ux_p[__w0] * __inl1_sz[__w0]));
        }
        for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
          uz[__w0] += ((__inl1_ux_p[__w0] * __inl1_sy[__w0]) - (__inl1_uy_p[__w0] * __inl1_sx[__w0]));
        }
        if (((mpt == 2) || (mpt == 0))) {
          for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
            ux[__w0] += (__inl1_econst * Ex[__w0]);
          }
          for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
            uy[__w0] += (__inl1_econst * Ey[__w0]);
          }
          for (int64_t __w0 = 0; __w0 < np_particles; ++__w0) {
            uz[__w0] += (__inl1_econst * Ez[__w0]);
          }
        }
        free(__cb1);
        free(__inl1_inv_gamma);
        free(__inl1_tx);
        free(__inl1_ty);
        free(__inl1_tz);
        free(__inl1_tsqi);
        free(__inl1_sx);
        free(__inl1_sy);
        free(__inl1_sz);
        free(__inl1_ux_p);
        free(__inl1_uy_p);
        free(__inl1_uz_p);
}

	void warpx_boris_push_fp64(const double *restrict Bx, const double *restrict By, const double *restrict Bz,
	                           const double *restrict Ex, const double *restrict Ey, const double *restrict Ez,
	                           double *restrict ux, double *restrict uy, double *restrict uz,
	                           const double m, const int64_t momentum_push_type,
	                           const int64_t np_particles, const double q) {
	    const double econst = (0.5 * q) * dt / m;
	    const double inv_c2 = 1.1126500560536185e-17;
	    const int64_t mpt = momentum_push_type;
	    #pragma omp parallel for
	    for (int64_t i = 0; i < np_particles; ++i) {
	        double ux_i = ux[i];
	        double uy_i = uy[i];
	        double uz_i = uz[i];
	        double Ex_i = Ex[i];
	        double Ey_i = Ey[i];
	        double Ez_i = Ez[i];
	        double Bx_i = Bx[i];
	        double By_i = By[i];
	        double Bz_i = Bz[i];
	        if (mpt == 1 || mpt == 0) {
	            ux_i += econst * Ex_i;
	            uy_i += econst * Ey_i;
	            uz_i += econst * Ez_i;
	        }
	        double inv_gamma = 1.0 / sqrt(1.0 + (ux_i*ux_i + uy_i*uy_i + uz_i*uz_i) * inv_c2);
	        double tx = econst * inv_gamma * Bx_i;
	        double ty = econst * inv_gamma * By_i;
	        double tz = econst * inv_gamma * Bz_i;
	        if (mpt == 1 || mpt == 2) {
	            double tsq = tx*tx + ty*ty + tz*tz;
	            if (tsq > 0.0) {
	                double factor = (sqrt(1.0 + tsq) - 1.0) / tsq;
	                tx *= factor;
	                ty *= factor;
	                tz *= factor;
	            } else {
	                double factor = 0.5;
	                tx *= factor;
	                ty *= factor;
	                tz *= factor;
	            }
	        }
	        double tsqi = 2.0 / (1.0 + tx*tx + ty*ty + tz*tz);
	        double sx = tx * tsqi;
	        double sy = ty * tsqi;
	        double sz = tz * tsqi;
	        double ux_p = ux_i + uy_i * tz - uz_i * ty;
	        double uy_p = uy_i + uz_i * tx - ux_i * tz;
	        double uz_p = uz_i + ux_i * ty - uy_i * tx;
	        ux_i += uy_p * sz - uz_p * sy;
	        uy_i += uz_p * sx - ux_p * sz;
	        uz_i += ux_p * sy - uy_p * sx;
	        if (mpt == 2 || mpt == 0) {
	            ux_i += econst * Ex_i;
	            uy_i += econst * Ey_i;
	            uz_i += econst * Ez_i;
	        }
	        ux[i] = ux_i;
	        uy[i] = uy_i;
	        uz[i] = uz_i;
	    }
	}
