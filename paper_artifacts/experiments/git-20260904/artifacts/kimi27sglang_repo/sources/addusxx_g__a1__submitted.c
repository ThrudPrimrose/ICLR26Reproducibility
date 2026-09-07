// hpcagent_bench-autogen -- generated from addusxx_g_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
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

static inline size_t align_up(size_t n, size_t align) {
    return (n + align - 1) & ~(align - 1);
}

void addusxx_g_fp64(const double _Complex *restrict becphi_c, const double _Complex *restrict becpsi_c, const double _Complex *restrict eigts1, const double _Complex *restrict eigts2, const double _Complex *restrict eigts3, const int64_t *restrict ijtoh, const int64_t *restrict ityp, const int64_t *restrict mill, const int64_t *restrict nh_type, const int64_t *restrict nij_type, const int64_t *restrict nl, const int64_t *restrict ofsbeta, const double _Complex *restrict qgm, double _Complex *restrict rhoc, const double *restrict tau, const int64_t *restrict tvanp, const double *restrict xk, const double *restrict xkq, const int64_t nat, const int64_t ngms, const int64_t nhm, const int64_t nij_tot, const int64_t nkb, const int64_t nnr, const int64_t nr1, const int64_t nr2, const int64_t nr3, const int64_t ntyp) {
    const double tpi = 2.0 * M_PI;

    double _Complex *eigqts = (double _Complex *)aligned_alloc(64, align_up((size_t)nat * sizeof(double _Complex), 64));
    for (int64_t na = 0; na < nat; ++na) {
        double s = 0.0;
        for (int d = 0; d < 3; ++d) {
            s += (xk[d] - xkq[d]) * tau[d * nat + na];
        }
        double arg = tpi * s;
        eigqts[na] = cos(arg) - (_Complex_I * sin(arg));
    }

    int64_t max_nh = 0;
    for (int64_t nt = 0; nt < ntyp; ++nt) {
        if (nh_type[nt] > max_nh) max_nh = nh_type[nt];
    }
    int64_t max_nij = max_nh * (max_nh + 1) / 2;

    double _Complex *Vt = (double _Complex *)aligned_alloc(64, align_up((size_t)max_nij * nat * sizeof(double _Complex), 64));
    double _Complex *phase_t = (double _Complex *)aligned_alloc(64, align_up((size_t)ngms * nat * sizeof(double _Complex), 64));
    int64_t *atoms = (int64_t *)aligned_alloc(64, align_up((size_t)nat * sizeof(int64_t), 64));

    for (int64_t nt = 0; nt < ntyp; ++nt) {
        if (!tvanp[nt]) continue;

        int64_t nh = nh_type[nt];
        int64_t nij = nh * (nh + 1) / 2;
        int64_t nij_off = nij_type[nt];

        int64_t n_a = 0;
        for (int64_t na = 0; na < nat; ++na) {
            if (ityp[na] == nt) atoms[n_a++] = na;
        }
        if (n_a == 0) continue;

        memset(Vt, 0, (size_t)nij * n_a * sizeof(double _Complex));
        for (int64_t a = 0; a < n_a; ++a) {
            int64_t na = atoms[a];
            int64_t ijkb0 = ofsbeta[na];
            double _Complex *vcol = Vt + a * nij;
            for (int64_t ih = 0; ih < nh; ++ih) {
                double _Complex ph = conj(becphi_c[ijkb0 + ih]);
                for (int64_t jh = 0; jh < nh; ++jh) {
                    int64_t k = ijtoh[(ih * nhm + jh) * ntyp + nt];
                    vcol[k] += becpsi_c[ijkb0 + jh] * ph;
                }
            }
        }

        for (int64_t ig = 0; ig < ngms; ++ig) {
            int64_t m0 = mill[0 * ngms + ig] + nr1;
            int64_t m1 = mill[1 * ngms + ig] + nr2;
            int64_t m2 = mill[2 * ngms + ig] + nr3;
            double _Complex *prow = phase_t + ig * n_a;
            for (int64_t a = 0; a < n_a; ++a) {
                int64_t na = atoms[a];
                prow[a] = eigqts[na] * eigts1[m0 * nat + na] * eigts2[m1 * nat + na] * eigts3[m2 * nat + na];
            }
        }

        #pragma omp parallel for schedule(static)
        for (int64_t ig = 0; ig < ngms; ++ig) {
            const double _Complex *qrow = qgm + ig * nij_tot + nij_off;
            const double _Complex *prow = phase_t + ig * n_a;
            double _Complex acc = 0.0;
            for (int64_t a = 0; a < n_a; ++a) {
                const double _Complex *vcol = Vt + a * nij;
                double _Complex dot = 0.0;
                for (int64_t k = 0; k < nij; ++k) {
                    dot += qrow[k] * vcol[k];
                }
                acc += dot * prow[a];
            }
            rhoc[nl[ig]] += acc;
        }
    }

    free(atoms);
    free(phase_t);
    free(Vt);
    free(eigqts);
}
