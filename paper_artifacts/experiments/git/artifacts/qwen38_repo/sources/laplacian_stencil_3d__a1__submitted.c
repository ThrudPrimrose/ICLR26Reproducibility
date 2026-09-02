// hpcagent_bench-autogen -- generated from laplacian_stencil_3d_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
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

void laplacian_stencil_3d_fp64(double *restrict ekin, double *restrict lap, const double *restrict psi, const int64_t N, const double inv_h2, const int64_t k) {
        int64_t m;
        double w;
        double *__cb1 = (double *)malloc((size_t)((N) * (N) * (N) * (k)) * sizeof(double));
        double *__cb2 = (double *)malloc((size_t)((N) * (N) * (N) * (k)) * sizeof(double));
        double *__cb3 = (double *)malloc((size_t)((N) * (N) * (N) * (k)) * sizeof(double));
        double *__cb4 = (double *)malloc((size_t)((N) * (N) * (N) * (k)) * sizeof(double));
        double *__cb5 = (double *)malloc((size_t)((N) * (N) * (N) * (k)) * sizeof(double));
        double *__cb6 = (double *)malloc((size_t)((N) * (N) * (N) * (k)) * sizeof(double));
        double *__cb7 = (double *)malloc((size_t)((N) * (N) * (N) * (k)) * sizeof(double));
        double *__cb8 = (double *)malloc((size_t)((N) * (N) * (N) * (k)) * sizeof(double));
        double *__cb9 = (double *)malloc((size_t)((N) * (N) * (N) * (k)) * sizeof(double));
        double *__cb10 = (double *)malloc((size_t)((N) * (N) * (N) * (k)) * sizeof(double));
        double *__cb11 = (double *)malloc((size_t)((N) * (N) * (N) * (k)) * sizeof(double));
        double *__cb12 = (double *)malloc((size_t)((N) * (N) * (N) * (k)) * sizeof(double));
        double *__cb13 = (double *)malloc((size_t)((N) * (N) * (N) * (k)) * sizeof(double));
        double *__cb14 = (double *)malloc((size_t)((N) * (N) * (N) * (k)) * sizeof(double));
        double *__cb15 = (double *)malloc((size_t)((N) * (N) * (N) * (k)) * sizeof(double));
        double *__cb16 = (double *)malloc((size_t)((N) * (N) * (N) * (k)) * sizeof(double));
        double *__cb17 = (double *)malloc((size_t)((N) * (N) * (N) * (k)) * sizeof(double));
        double *__cb18 = (double *)malloc((size_t)((N) * (N) * (N) * (k)) * sizeof(double));
        double *__cb19 = (double *)malloc((size_t)((N) * (N) * (N) * (k)) * sizeof(double));
        double *__cb20 = (double *)malloc((size_t)((N) * (N) * (N) * (k)) * sizeof(double));
        double *__cb21 = (double *)malloc((size_t)((N) * (N) * (N) * (k)) * sizeof(double));
        double *__cb22 = (double *)malloc((size_t)((N) * (N) * (N) * (k)) * sizeof(double));
        double *__cb23 = (double *)malloc((size_t)((N) * (N) * (N) * (k)) * sizeof(double));
        double *__cb24 = (double *)malloc((size_t)((N) * (N) * (N) * (k)) * sizeof(double));
        double *__cb25 = (double *)malloc((size_t)((k)) * sizeof(double));
        double *acc = (double *)malloc((size_t)((N) * (N) * (N) * (k)) * sizeof(double));
        for (int64_t __w0 = 0; __w0 < N; ++__w0) {
          for (int64_t __w1 = 0; __w1 < N; ++__w1) {
            for (int64_t __w2 = 0; __w2 < N; ++__w2) {
              for (int64_t __w3 = 0; __w3 < k; ++__w3) {
                acc[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] = ((3.0 * -2.8472222222222223) * psi[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)]);
              }
            }
          }
        }
        m = (1 + 0);
        w = 1.6;
        /* numpy: np.roll(psi, m, axis=0) */
        for (int64_t __rl0 = 0; __rl0 < N; ++__rl0) {
          for (int64_t __rl1 = 0; __rl1 < N; ++__rl1) {
            for (int64_t __rl2 = 0; __rl2 < N; ++__rl2) {
              for (int64_t __rl3 = 0; __rl3 < k; ++__rl3) {
                __cb1[(((__rl0)*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)] = psi[(((python_mod((python_mod((__rl0 - m), N) + N), N))*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)];
              }
            }
          }
        }
        /* numpy: np.roll(psi, -m, axis=0) */
        for (int64_t __rl0 = 0; __rl0 < N; ++__rl0) {
          for (int64_t __rl1 = 0; __rl1 < N; ++__rl1) {
            for (int64_t __rl2 = 0; __rl2 < N; ++__rl2) {
              for (int64_t __rl3 = 0; __rl3 < k; ++__rl3) {
                __cb2[(((__rl0)*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)] = psi[(((python_mod((python_mod((__rl0 - (-m)), N) + N), N))*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)];
              }
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < N; ++__w0) {
          for (int64_t __w1 = 0; __w1 < N; ++__w1) {
            for (int64_t __w2 = 0; __w2 < N; ++__w2) {
              for (int64_t __w3 = 0; __w3 < k; ++__w3) {
                acc[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] = (acc[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] + (w * (__cb1[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] + __cb2[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)])));
              }
            }
          }
        }
        m = (1 + 1);
        w = (-0.2);
        /* numpy: np.roll(psi, m, axis=0) */
        for (int64_t __rl0 = 0; __rl0 < N; ++__rl0) {
          for (int64_t __rl1 = 0; __rl1 < N; ++__rl1) {
            for (int64_t __rl2 = 0; __rl2 < N; ++__rl2) {
              for (int64_t __rl3 = 0; __rl3 < k; ++__rl3) {
                __cb3[(((__rl0)*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)] = psi[(((python_mod((python_mod((__rl0 - m), N) + N), N))*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)];
              }
            }
          }
        }
        /* numpy: np.roll(psi, -m, axis=0) */
        for (int64_t __rl0 = 0; __rl0 < N; ++__rl0) {
          for (int64_t __rl1 = 0; __rl1 < N; ++__rl1) {
            for (int64_t __rl2 = 0; __rl2 < N; ++__rl2) {
              for (int64_t __rl3 = 0; __rl3 < k; ++__rl3) {
                __cb4[(((__rl0)*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)] = psi[(((python_mod((python_mod((__rl0 - (-m)), N) + N), N))*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)];
              }
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < N; ++__w0) {
          for (int64_t __w1 = 0; __w1 < N; ++__w1) {
            for (int64_t __w2 = 0; __w2 < N; ++__w2) {
              for (int64_t __w3 = 0; __w3 < k; ++__w3) {
                acc[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] = (acc[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] + (w * (__cb3[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] + __cb4[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)])));
              }
            }
          }
        }
        m = (1 + 2);
        w = 0.025396825396825397;
        /* numpy: np.roll(psi, m, axis=0) */
        for (int64_t __rl0 = 0; __rl0 < N; ++__rl0) {
          for (int64_t __rl1 = 0; __rl1 < N; ++__rl1) {
            for (int64_t __rl2 = 0; __rl2 < N; ++__rl2) {
              for (int64_t __rl3 = 0; __rl3 < k; ++__rl3) {
                __cb5[(((__rl0)*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)] = psi[(((python_mod((python_mod((__rl0 - m), N) + N), N))*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)];
              }
            }
          }
        }
        /* numpy: np.roll(psi, -m, axis=0) */
        for (int64_t __rl0 = 0; __rl0 < N; ++__rl0) {
          for (int64_t __rl1 = 0; __rl1 < N; ++__rl1) {
            for (int64_t __rl2 = 0; __rl2 < N; ++__rl2) {
              for (int64_t __rl3 = 0; __rl3 < k; ++__rl3) {
                __cb6[(((__rl0)*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)] = psi[(((python_mod((python_mod((__rl0 - (-m)), N) + N), N))*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)];
              }
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < N; ++__w0) {
          for (int64_t __w1 = 0; __w1 < N; ++__w1) {
            for (int64_t __w2 = 0; __w2 < N; ++__w2) {
              for (int64_t __w3 = 0; __w3 < k; ++__w3) {
                acc[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] = (acc[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] + (w * (__cb5[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] + __cb6[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)])));
              }
            }
          }
        }
        m = (1 + 3);
        w = (-0.0017857142857142857);
        /* numpy: np.roll(psi, m, axis=0) */
        for (int64_t __rl0 = 0; __rl0 < N; ++__rl0) {
          for (int64_t __rl1 = 0; __rl1 < N; ++__rl1) {
            for (int64_t __rl2 = 0; __rl2 < N; ++__rl2) {
              for (int64_t __rl3 = 0; __rl3 < k; ++__rl3) {
                __cb7[(((__rl0)*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)] = psi[(((python_mod((python_mod((__rl0 - m), N) + N), N))*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)];
              }
            }
          }
        }
        /* numpy: np.roll(psi, -m, axis=0) */
        for (int64_t __rl0 = 0; __rl0 < N; ++__rl0) {
          for (int64_t __rl1 = 0; __rl1 < N; ++__rl1) {
            for (int64_t __rl2 = 0; __rl2 < N; ++__rl2) {
              for (int64_t __rl3 = 0; __rl3 < k; ++__rl3) {
                __cb8[(((__rl0)*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)] = psi[(((python_mod((python_mod((__rl0 - (-m)), N) + N), N))*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)];
              }
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < N; ++__w0) {
          for (int64_t __w1 = 0; __w1 < N; ++__w1) {
            for (int64_t __w2 = 0; __w2 < N; ++__w2) {
              for (int64_t __w3 = 0; __w3 < k; ++__w3) {
                acc[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] = (acc[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] + (w * (__cb7[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] + __cb8[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)])));
              }
            }
          }
        }
        m = (1 + 0);
        w = 1.6;
        /* numpy: np.roll(psi, m, axis=1) */
        for (int64_t __rl0 = 0; __rl0 < N; ++__rl0) {
          for (int64_t __rl1 = 0; __rl1 < N; ++__rl1) {
            for (int64_t __rl2 = 0; __rl2 < N; ++__rl2) {
              for (int64_t __rl3 = 0; __rl3 < k; ++__rl3) {
                __cb9[(((__rl0)*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)] = psi[(((__rl0)*(N) + (python_mod((python_mod((__rl1 - m), N) + N), N)))*(N) + (__rl2))*(k) + (__rl3)];
              }
            }
          }
        }
        /* numpy: np.roll(psi, -m, axis=1) */
        for (int64_t __rl0 = 0; __rl0 < N; ++__rl0) {
          for (int64_t __rl1 = 0; __rl1 < N; ++__rl1) {
            for (int64_t __rl2 = 0; __rl2 < N; ++__rl2) {
              for (int64_t __rl3 = 0; __rl3 < k; ++__rl3) {
                __cb10[(((__rl0)*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)] = psi[(((__rl0)*(N) + (python_mod((python_mod((__rl1 - (-m)), N) + N), N)))*(N) + (__rl2))*(k) + (__rl3)];
              }
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < N; ++__w0) {
          for (int64_t __w1 = 0; __w1 < N; ++__w1) {
            for (int64_t __w2 = 0; __w2 < N; ++__w2) {
              for (int64_t __w3 = 0; __w3 < k; ++__w3) {
                acc[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] = (acc[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] + (w * (__cb9[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] + __cb10[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)])));
              }
            }
          }
        }
        m = (1 + 1);
        w = (-0.2);
        /* numpy: np.roll(psi, m, axis=1) */
        for (int64_t __rl0 = 0; __rl0 < N; ++__rl0) {
          for (int64_t __rl1 = 0; __rl1 < N; ++__rl1) {
            for (int64_t __rl2 = 0; __rl2 < N; ++__rl2) {
              for (int64_t __rl3 = 0; __rl3 < k; ++__rl3) {
                __cb11[(((__rl0)*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)] = psi[(((__rl0)*(N) + (python_mod((python_mod((__rl1 - m), N) + N), N)))*(N) + (__rl2))*(k) + (__rl3)];
              }
            }
          }
        }
        /* numpy: np.roll(psi, -m, axis=1) */
        for (int64_t __rl0 = 0; __rl0 < N; ++__rl0) {
          for (int64_t __rl1 = 0; __rl1 < N; ++__rl1) {
            for (int64_t __rl2 = 0; __rl2 < N; ++__rl2) {
              for (int64_t __rl3 = 0; __rl3 < k; ++__rl3) {
                __cb12[(((__rl0)*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)] = psi[(((__rl0)*(N) + (python_mod((python_mod((__rl1 - (-m)), N) + N), N)))*(N) + (__rl2))*(k) + (__rl3)];
              }
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < N; ++__w0) {
          for (int64_t __w1 = 0; __w1 < N; ++__w1) {
            for (int64_t __w2 = 0; __w2 < N; ++__w2) {
              for (int64_t __w3 = 0; __w3 < k; ++__w3) {
                acc[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] = (acc[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] + (w * (__cb11[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] + __cb12[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)])));
              }
            }
          }
        }
        m = (1 + 2);
        w = 0.025396825396825397;
        /* numpy: np.roll(psi, m, axis=1) */
        for (int64_t __rl0 = 0; __rl0 < N; ++__rl0) {
          for (int64_t __rl1 = 0; __rl1 < N; ++__rl1) {
            for (int64_t __rl2 = 0; __rl2 < N; ++__rl2) {
              for (int64_t __rl3 = 0; __rl3 < k; ++__rl3) {
                __cb13[(((__rl0)*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)] = psi[(((__rl0)*(N) + (python_mod((python_mod((__rl1 - m), N) + N), N)))*(N) + (__rl2))*(k) + (__rl3)];
              }
            }
          }
        }
        /* numpy: np.roll(psi, -m, axis=1) */
        for (int64_t __rl0 = 0; __rl0 < N; ++__rl0) {
          for (int64_t __rl1 = 0; __rl1 < N; ++__rl1) {
            for (int64_t __rl2 = 0; __rl2 < N; ++__rl2) {
              for (int64_t __rl3 = 0; __rl3 < k; ++__rl3) {
                __cb14[(((__rl0)*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)] = psi[(((__rl0)*(N) + (python_mod((python_mod((__rl1 - (-m)), N) + N), N)))*(N) + (__rl2))*(k) + (__rl3)];
              }
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < N; ++__w0) {
          for (int64_t __w1 = 0; __w1 < N; ++__w1) {
            for (int64_t __w2 = 0; __w2 < N; ++__w2) {
              for (int64_t __w3 = 0; __w3 < k; ++__w3) {
                acc[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] = (acc[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] + (w * (__cb13[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] + __cb14[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)])));
              }
            }
          }
        }
        m = (1 + 3);
        w = (-0.0017857142857142857);
        /* numpy: np.roll(psi, m, axis=1) */
        for (int64_t __rl0 = 0; __rl0 < N; ++__rl0) {
          for (int64_t __rl1 = 0; __rl1 < N; ++__rl1) {
            for (int64_t __rl2 = 0; __rl2 < N; ++__rl2) {
              for (int64_t __rl3 = 0; __rl3 < k; ++__rl3) {
                __cb15[(((__rl0)*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)] = psi[(((__rl0)*(N) + (python_mod((python_mod((__rl1 - m), N) + N), N)))*(N) + (__rl2))*(k) + (__rl3)];
              }
            }
          }
        }
        /* numpy: np.roll(psi, -m, axis=1) */
        for (int64_t __rl0 = 0; __rl0 < N; ++__rl0) {
          for (int64_t __rl1 = 0; __rl1 < N; ++__rl1) {
            for (int64_t __rl2 = 0; __rl2 < N; ++__rl2) {
              for (int64_t __rl3 = 0; __rl3 < k; ++__rl3) {
                __cb16[(((__rl0)*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)] = psi[(((__rl0)*(N) + (python_mod((python_mod((__rl1 - (-m)), N) + N), N)))*(N) + (__rl2))*(k) + (__rl3)];
              }
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < N; ++__w0) {
          for (int64_t __w1 = 0; __w1 < N; ++__w1) {
            for (int64_t __w2 = 0; __w2 < N; ++__w2) {
              for (int64_t __w3 = 0; __w3 < k; ++__w3) {
                acc[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] = (acc[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] + (w * (__cb15[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] + __cb16[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)])));
              }
            }
          }
        }
        m = (1 + 0);
        w = 1.6;
        /* numpy: np.roll(psi, m, axis=2) */
        for (int64_t __rl0 = 0; __rl0 < N; ++__rl0) {
          for (int64_t __rl1 = 0; __rl1 < N; ++__rl1) {
            for (int64_t __rl2 = 0; __rl2 < N; ++__rl2) {
              for (int64_t __rl3 = 0; __rl3 < k; ++__rl3) {
                __cb17[(((__rl0)*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)] = psi[(((__rl0)*(N) + (__rl1))*(N) + (python_mod((python_mod((__rl2 - m), N) + N), N)))*(k) + (__rl3)];
              }
            }
          }
        }
        /* numpy: np.roll(psi, -m, axis=2) */
        for (int64_t __rl0 = 0; __rl0 < N; ++__rl0) {
          for (int64_t __rl1 = 0; __rl1 < N; ++__rl1) {
            for (int64_t __rl2 = 0; __rl2 < N; ++__rl2) {
              for (int64_t __rl3 = 0; __rl3 < k; ++__rl3) {
                __cb18[(((__rl0)*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)] = psi[(((__rl0)*(N) + (__rl1))*(N) + (python_mod((python_mod((__rl2 - (-m)), N) + N), N)))*(k) + (__rl3)];
              }
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < N; ++__w0) {
          for (int64_t __w1 = 0; __w1 < N; ++__w1) {
            for (int64_t __w2 = 0; __w2 < N; ++__w2) {
              for (int64_t __w3 = 0; __w3 < k; ++__w3) {
                acc[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] = (acc[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] + (w * (__cb17[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] + __cb18[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)])));
              }
            }
          }
        }
        m = (1 + 1);
        w = (-0.2);
        /* numpy: np.roll(psi, m, axis=2) */
        for (int64_t __rl0 = 0; __rl0 < N; ++__rl0) {
          for (int64_t __rl1 = 0; __rl1 < N; ++__rl1) {
            for (int64_t __rl2 = 0; __rl2 < N; ++__rl2) {
              for (int64_t __rl3 = 0; __rl3 < k; ++__rl3) {
                __cb19[(((__rl0)*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)] = psi[(((__rl0)*(N) + (__rl1))*(N) + (python_mod((python_mod((__rl2 - m), N) + N), N)))*(k) + (__rl3)];
              }
            }
          }
        }
        /* numpy: np.roll(psi, -m, axis=2) */
        for (int64_t __rl0 = 0; __rl0 < N; ++__rl0) {
          for (int64_t __rl1 = 0; __rl1 < N; ++__rl1) {
            for (int64_t __rl2 = 0; __rl2 < N; ++__rl2) {
              for (int64_t __rl3 = 0; __rl3 < k; ++__rl3) {
                __cb20[(((__rl0)*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)] = psi[(((__rl0)*(N) + (__rl1))*(N) + (python_mod((python_mod((__rl2 - (-m)), N) + N), N)))*(k) + (__rl3)];
              }
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < N; ++__w0) {
          for (int64_t __w1 = 0; __w1 < N; ++__w1) {
            for (int64_t __w2 = 0; __w2 < N; ++__w2) {
              for (int64_t __w3 = 0; __w3 < k; ++__w3) {
                acc[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] = (acc[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] + (w * (__cb19[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] + __cb20[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)])));
              }
            }
          }
        }
        m = (1 + 2);
        w = 0.025396825396825397;
        /* numpy: np.roll(psi, m, axis=2) */
        for (int64_t __rl0 = 0; __rl0 < N; ++__rl0) {
          for (int64_t __rl1 = 0; __rl1 < N; ++__rl1) {
            for (int64_t __rl2 = 0; __rl2 < N; ++__rl2) {
              for (int64_t __rl3 = 0; __rl3 < k; ++__rl3) {
                __cb21[(((__rl0)*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)] = psi[(((__rl0)*(N) + (__rl1))*(N) + (python_mod((python_mod((__rl2 - m), N) + N), N)))*(k) + (__rl3)];
              }
            }
          }
        }
        /* numpy: np.roll(psi, -m, axis=2) */
        for (int64_t __rl0 = 0; __rl0 < N; ++__rl0) {
          for (int64_t __rl1 = 0; __rl1 < N; ++__rl1) {
            for (int64_t __rl2 = 0; __rl2 < N; ++__rl2) {
              for (int64_t __rl3 = 0; __rl3 < k; ++__rl3) {
                __cb22[(((__rl0)*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)] = psi[(((__rl0)*(N) + (__rl1))*(N) + (python_mod((python_mod((__rl2 - (-m)), N) + N), N)))*(k) + (__rl3)];
              }
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < N; ++__w0) {
          for (int64_t __w1 = 0; __w1 < N; ++__w1) {
            for (int64_t __w2 = 0; __w2 < N; ++__w2) {
              for (int64_t __w3 = 0; __w3 < k; ++__w3) {
                acc[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] = (acc[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] + (w * (__cb21[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] + __cb22[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)])));
              }
            }
          }
        }
        m = (1 + 3);
        w = (-0.0017857142857142857);
        /* numpy: np.roll(psi, m, axis=2) */
        for (int64_t __rl0 = 0; __rl0 < N; ++__rl0) {
          for (int64_t __rl1 = 0; __rl1 < N; ++__rl1) {
            for (int64_t __rl2 = 0; __rl2 < N; ++__rl2) {
              for (int64_t __rl3 = 0; __rl3 < k; ++__rl3) {
                __cb23[(((__rl0)*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)] = psi[(((__rl0)*(N) + (__rl1))*(N) + (python_mod((python_mod((__rl2 - m), N) + N), N)))*(k) + (__rl3)];
              }
            }
          }
        }
        /* numpy: np.roll(psi, -m, axis=2) */
        for (int64_t __rl0 = 0; __rl0 < N; ++__rl0) {
          for (int64_t __rl1 = 0; __rl1 < N; ++__rl1) {
            for (int64_t __rl2 = 0; __rl2 < N; ++__rl2) {
              for (int64_t __rl3 = 0; __rl3 < k; ++__rl3) {
                __cb24[(((__rl0)*(N) + (__rl1))*(N) + (__rl2))*(k) + (__rl3)] = psi[(((__rl0)*(N) + (__rl1))*(N) + (python_mod((python_mod((__rl2 - (-m)), N) + N), N)))*(k) + (__rl3)];
              }
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < N; ++__w0) {
          for (int64_t __w1 = 0; __w1 < N; ++__w1) {
            for (int64_t __w2 = 0; __w2 < N; ++__w2) {
              for (int64_t __w3 = 0; __w3 < k; ++__w3) {
                acc[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] = (acc[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] + (w * (__cb23[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)] + __cb24[(((__w0)*(N) + (__w1))*(N) + (__w2))*(k) + (__w3)])));
              }
            }
          }
        }
        for (int64_t si0 = 0; si0 < N; ++si0) {
          for (int64_t si1 = 0; si1 < N; ++si1) {
            for (int64_t si2 = 0; si2 < N; ++si2) {
              for (int64_t si3 = 0; si3 < k; ++si3) {
                lap[(((si0)*(N) + (si1))*(N) + (si2))*(k) + (si3)] = (inv_h2 * acc[(((si0)*(N) + (si1))*(N) + (si2))*(k) + (si3)]);
              }
            }
          }
        }
        /* numpy: np.einsum('xyzk,xyzk->k', psi, lap) */
        for (int64_t __es_k = 0; __es_k < k; ++__es_k) {
          __cb25[__es_k] = 0.0;
          for (int64_t __es_x = 0; __es_x < N; ++__es_x) {
            for (int64_t __es_y = 0; __es_y < N; ++__es_y) {
              for (int64_t __es_z = 0; __es_z < N; ++__es_z) {
                __cb25[__es_k] += (psi[(((__es_x)*(N) + (__es_y))*(N) + (__es_z))*(k) + (__es_k)] * lap[(((__es_x)*(N) + (__es_y))*(N) + (__es_z))*(k) + (__es_k)]);
              }
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < k; ++__w0) {
          ekin[__w0] = ((-0.5) * __cb25[__w0]);
        }
        free(__cb1);
        free(__cb2);
        free(__cb3);
        free(__cb4);
        free(__cb5);
        free(__cb6);
        free(__cb7);
        free(__cb8);
        free(__cb9);
        free(__cb10);
        free(__cb11);
        free(__cb12);
        free(__cb13);
        free(__cb14);
        free(__cb15);
        free(__cb16);
        free(__cb17);
        free(__cb18);
        free(__cb19);
        free(__cb20);
        free(__cb21);
        free(__cb22);
        free(__cb23);
        free(__cb24);
        free(__cb25);
        free(acc);
}
