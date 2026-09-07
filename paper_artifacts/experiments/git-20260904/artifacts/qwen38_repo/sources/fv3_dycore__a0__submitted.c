// hpcagent_bench-autogen -- generated from fv3_dycore_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
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

constexpr int64_t nhalo = 3;

void fv3_dycore_hord5_gt3_fp64(const double *restrict area, const double *restrict crx, const double *restrict cry, const double *restrict dxa, const double *restrict dya, double *restrict q, double *restrict q_x_flux, double *restrict q_y_flux, const double *restrict x_area_flux, const double *restrict y_area_flux, const int64_t grid_type, const int64_t hord, const int64_t ni, const int64_t nj, const int64_t nk, uint8_t *restrict workspace, const int64_t workspace_size) {
    (void)workspace; (void)workspace_size;
        int64_t __inl1_nx;
        int64_t __inl1_ny;
        int64_t __inl1_ord_inner;
        int64_t __inl13_j_end;
        int64_t __inl13_lo;
        int64_t __inl13_hi;
        int64_t __inl14_j_end;
        int64_t __inl14_lo;
        int64_t __inl14_hi;
        int64_t __inl4_ny;
        int64_t __inl4_j0;
        int64_t __inl4_j1;
        int64_t __inl15_i_end;
        int64_t __inl15_lo;
        int64_t __inl15_hi;
        int64_t __inl16_i_end;
        int64_t __inl16_lo;
        int64_t __inl16_hi;
        int64_t __inl17_i_end;
        int64_t __inl17_lo;
        int64_t __inl17_hi;
        int64_t __inl18_i_end;
        int64_t __inl18_lo;
        int64_t __inl18_hi;
        int64_t __inl8_nx;
        int64_t __inl8_i0;
        int64_t __inl8_i1;
        int64_t __inl19_j_end;
        int64_t __inl19_lo;
        int64_t __inl19_hi;
        int64_t __inl20_j_end;
        int64_t __inl20_lo;
        int64_t __inl20_hi;
        int64_t __inl10_i_end;
        int64_t __inl10_j_end;
        double *__inl1_q_y_advected_mean = (double *)malloc((size_t)((((2 * nhalo) + ni)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        memset(__inl1_q_y_advected_mean, 0, (size_t)((((2 * nhalo) + ni)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl1_q_x_advected_mean = (double *)malloc((size_t)((((2 * nhalo) + ni)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        memset(__inl1_q_x_advected_mean, 0, (size_t)((((2 * nhalo) + ni)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl1_q_advected_y = (double *)malloc((size_t)((((2 * nhalo) + ni)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        memset(__inl1_q_advected_y, 0, (size_t)((((2 * nhalo) + ni)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl1_q_advected_x = (double *)malloc((size_t)((((2 * nhalo) + ni)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        memset(__inl1_q_advected_x, 0, (size_t)((((2 * nhalo) + ni)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl1_q_ayxa = (double *)malloc((size_t)((((2 * nhalo) + ni)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        memset(__inl1_q_ayxa, 0, (size_t)((((2 * nhalo) + ni)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl1_q_axya = (double *)malloc((size_t)((((2 * nhalo) + ni)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        memset(__inl1_q_axya, 0, (size_t)((((2 * nhalo) + ni)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl1_al = (double *)malloc((size_t)((((2 * nhalo) + ni)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        memset(__inl1_al, 0, (size_t)((((2 * nhalo) + ni)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__cb1 = (double *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(double));
        double *__cb2 = (double *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__cb3 = (double *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__cb4 = (double *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(double));
        double *__inl14_bl = (double *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(double));
        double *__inl14_br = (double *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(double));
        double *__inl14_b0 = (double *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(double));
        double *__inl14_bl_m1 = (double *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(double));
        double *__inl14_br_m1 = (double *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(double));
        double *__inl14_b0_m1 = (double *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(double));
        bool *__inl14_smt5 = (bool *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(bool));
        bool *__inl14_smt5_m1 = (bool *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(bool));
        double *__inl4_fyy_j = (double *)malloc((size_t)((((2 * nhalo) + ni)) * ((((2 * nhalo) + nj) - 6)) * (nk)) * sizeof(double));
        double *__inl4_fyy_jp1 = (double *)malloc((size_t)((((2 * nhalo) + ni)) * ((((2 * nhalo) + nj) - 6)) * (nk)) * sizeof(double));
        double *__inl4_denom = (double *)malloc((size_t)((((2 * nhalo) + ni)) * ((((2 * nhalo) + nj) - 6)) * (nk)) * sizeof(double));
        double *__inl16_bl = (double *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl16_br = (double *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl16_b0 = (double *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl16_bl_m1 = (double *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl16_br_m1 = (double *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl16_b0_m1 = (double *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        bool *__inl16_smt5 = (bool *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(bool));
        bool *__inl16_smt5_m1 = (bool *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(bool));
        double *__inl18_bl = (double *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl18_br = (double *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl18_b0 = (double *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl18_bl_m1 = (double *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl18_br_m1 = (double *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl18_b0_m1 = (double *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        bool *__inl18_smt5 = (bool *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(bool));
        bool *__inl18_smt5_m1 = (bool *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(bool));
        double *__inl8_fx1_i = (double *)malloc((size_t)(((((2 * nhalo) + ni) - 6)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl8_fx1_ip1 = (double *)malloc((size_t)(((((2 * nhalo) + ni) - 6)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl8_area_with_x_flux = (double *)malloc((size_t)(((((2 * nhalo) + ni) - 6)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl20_bl = (double *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(double));
        double *__inl20_br = (double *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(double));
        double *__inl20_b0 = (double *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(double));
        double *__inl20_bl_m1 = (double *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(double));
        double *__inl20_br_m1 = (double *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(double));
        double *__inl20_b0_m1 = (double *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(double));
        bool *__inl20_smt5 = (bool *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(bool));
        bool *__inl20_smt5_m1 = (bool *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(bool));
        double *__inl1_xuf = (double *)malloc((size_t)((((2 * nhalo) + ni)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl1_yuf = (double *)malloc((size_t)((((2 * nhalo) + ni)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl14_c = (double *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(double));
        double *__inl14_q_j = (double *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(double));
        double *__inl14_q_jm1 = (double *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(double));
        double *__inl16_c = (double *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl16_q_i = (double *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl16_q_im1 = (double *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl18_c = (double *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl18_q_i = (double *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl18_q_im1 = (double *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl20_c = (double *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(double));
        double *__inl20_q_j = (double *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(double));
        double *__inl20_q_jm1 = (double *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(double));
        double *__inl14_mask = (double *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(double));
        double *__inl16_mask = (double *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl18_mask = (double *)malloc((size_t)(((ni + 1)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        double *__inl20_mask = (double *)malloc((size_t)((((2 * nhalo) + ni)) * ((nj + 1)) * (nk)) * sizeof(double));
        __inl1_nx = ((nhalo + ni) + nhalo);
        __inl1_ny = ((nhalo + nj) + nhalo);
        __inl1_ord_inner = ((hord == 10) ? 8 : hord);
        memset(__inl1_q_y_advected_mean, 0, (size_t)((((2 * nhalo) + ni)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        memset(__inl1_q_x_advected_mean, 0, (size_t)((((2 * nhalo) + ni)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        memset(__inl1_q_advected_y, 0, (size_t)((((2 * nhalo) + ni)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        memset(__inl1_q_advected_x, 0, (size_t)((((2 * nhalo) + ni)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        memset(__inl1_q_ayxa, 0, (size_t)((((2 * nhalo) + ni)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        memset(__inl1_q_axya, 0, (size_t)((((2 * nhalo) + ni)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        memset(__inl1_al, 0, (size_t)((((2 * nhalo) + ni)) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((0)*(((nhalo + nj) + nhalo)) + (0))*(nk) + (si2)] = q[((5)*(((nhalo + nj) + nhalo)) + (0))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((1)*(((nhalo + nj) + nhalo)) + (0))*(nk) + (si2)] = q[((5)*(((nhalo + nj) + nhalo)) + (1))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((2)*(((nhalo + nj) + nhalo)) + (0))*(nk) + (si2)] = q[((5)*(((nhalo + nj) + nhalo)) + (2))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((0)*(((nhalo + nj) + nhalo)) + (1))*(nk) + (si2)] = q[((4)*(((nhalo + nj) + nhalo)) + (0))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((1)*(((nhalo + nj) + nhalo)) + (1))*(nk) + (si2)] = q[((4)*(((nhalo + nj) + nhalo)) + (1))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((2)*(((nhalo + nj) + nhalo)) + (1))*(nk) + (si2)] = q[((4)*(((nhalo + nj) + nhalo)) + (2))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((0)*(((nhalo + nj) + nhalo)) + (2))*(nk) + (si2)] = q[((3)*(((nhalo + nj) + nhalo)) + (0))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((1)*(((nhalo + nj) + nhalo)) + (2))*(nk) + (si2)] = q[((3)*(((nhalo + nj) + nhalo)) + (1))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((2)*(((nhalo + nj) + nhalo)) + (2))*(nk) + (si2)] = q[((3)*(((nhalo + nj) + nhalo)) + (2))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 4))*(((nhalo + nj) + nhalo)) + (0))*(nk) + (si2)] = q[(((2 * nhalo + ni - 7))*(((nhalo + nj) + nhalo)) + (2))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 3))*(((nhalo + nj) + nhalo)) + (0))*(nk) + (si2)] = q[(((2 * nhalo + ni - 7))*(((nhalo + nj) + nhalo)) + (1))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 2))*(((nhalo + nj) + nhalo)) + (0))*(nk) + (si2)] = q[(((2 * nhalo + ni - 7))*(((nhalo + nj) + nhalo)) + (0))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 4))*(((nhalo + nj) + nhalo)) + (1))*(nk) + (si2)] = q[(((2 * nhalo + ni - 6))*(((nhalo + nj) + nhalo)) + (2))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 3))*(((nhalo + nj) + nhalo)) + (1))*(nk) + (si2)] = q[(((2 * nhalo + ni - 6))*(((nhalo + nj) + nhalo)) + (1))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 2))*(((nhalo + nj) + nhalo)) + (1))*(nk) + (si2)] = q[(((2 * nhalo + ni - 6))*(((nhalo + nj) + nhalo)) + (0))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 4))*(((nhalo + nj) + nhalo)) + (2))*(nk) + (si2)] = q[(((2 * nhalo + ni - 5))*(((nhalo + nj) + nhalo)) + (2))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 3))*(((nhalo + nj) + nhalo)) + (2))*(nk) + (si2)] = q[(((2 * nhalo + ni - 5))*(((nhalo + nj) + nhalo)) + (1))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 2))*(((nhalo + nj) + nhalo)) + (2))*(nk) + (si2)] = q[(((2 * nhalo + ni - 5))*(((nhalo + nj) + nhalo)) + (0))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((0)*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 2)))*(nk) + (si2)] = q[((5)*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 2)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((0)*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 3)))*(nk) + (si2)] = q[((4)*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 2)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((0)*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 4)))*(nk) + (si2)] = q[((3)*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 2)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((1)*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 2)))*(nk) + (si2)] = q[((5)*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 3)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((1)*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 3)))*(nk) + (si2)] = q[((4)*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 3)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((1)*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 4)))*(nk) + (si2)] = q[((3)*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 3)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((2)*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 2)))*(nk) + (si2)] = q[((5)*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 4)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((2)*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 3)))*(nk) + (si2)] = q[((4)*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 4)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((2)*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 4)))*(nk) + (si2)] = q[((3)*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 4)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 2))*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 4)))*(nk) + (si2)] = q[(((2 * nhalo + ni - 5))*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 2)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 2))*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 3)))*(nk) + (si2)] = q[(((2 * nhalo + ni - 6))*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 2)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 2))*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 2)))*(nk) + (si2)] = q[(((2 * nhalo + ni - 7))*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 2)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 3))*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 4)))*(nk) + (si2)] = q[(((2 * nhalo + ni - 5))*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 3)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 3))*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 3)))*(nk) + (si2)] = q[(((2 * nhalo + ni - 6))*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 3)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 3))*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 2)))*(nk) + (si2)] = q[(((2 * nhalo + ni - 7))*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 3)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 4))*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 4)))*(nk) + (si2)] = q[(((2 * nhalo + ni - 5))*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 4)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 4))*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 3)))*(nk) + (si2)] = q[(((2 * nhalo + ni - 6))*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 4)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 4))*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 2)))*(nk) + (si2)] = q[(((2 * nhalo + ni - 7))*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 4)))*(nk) + (si2)];
        }
        __inl13_j_end = ((nhalo + nj) - 1);
        __inl13_lo = (nhalo - 1);
        __inl13_hi = (__inl13_j_end + 3);
        for (int64_t si0 = 0; si0 < ((2 * nhalo) + ni); ++si0) {
          for (int64_t si1 = __inl13_lo; si1 < __inl13_hi; ++si1) {
            for (int64_t si2 = 0; si2 < nk; ++si2) {
              __inl1_al[((si0)*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)] = ((0.5833333333333334 * (q[((si0)*(((nhalo + nj) + nhalo)) + ((si1 + ((__inl13_lo - 1) - __inl13_lo))))*(nk) + (si2)] + q[((si0)*(((nhalo + nj) + nhalo)) + ((si1 + (__inl13_lo - __inl13_lo))))*(nk) + (si2)])) + (-0.08333333333333333 * (q[((si0)*(((nhalo + nj) + nhalo)) + ((si1 + ((__inl13_lo - 2) - __inl13_lo))))*(nk) + (si2)] + q[((si0)*(((nhalo + nj) + nhalo)) + ((si1 + ((__inl13_lo + 1) - __inl13_lo))))*(nk) + (si2)])));
            }
          }
        }
        if ((grid_type < 3)) {
          int64_t *__inl13_ja = (int64_t *)malloc(((2)) * sizeof(int64_t));
          __inl13_ja[0] = (nhalo - 1);
          __inl13_ja[1] = __inl13_j_end;
          for (int64_t __sc0 = 0; __sc0 < 2; ++__sc0) {
            for (int64_t __scs0 = 0; __scs0 < __inl1_nx; ++__scs0) {
              for (int64_t __scs1 = 0; __scs1 < nk; ++__scs1) {
                __inl1_al[((__scs0)*(((2 * nhalo) + nj)) + (__inl13_ja[__sc0]))*(nk) + (__scs1)] = (((-0.14285714285714285 * q[((__scs0)*(((nhalo + nj) + nhalo)) + ((__inl13_ja[__sc0] - 2)))*(nk) + (__scs1)]) + (0.7857142857142857 * q[((__scs0)*(((nhalo + nj) + nhalo)) + ((__inl13_ja[__sc0] - 1)))*(nk) + (__scs1)])) + (0.35714285714285715 * q[((__scs0)*(((nhalo + nj) + nhalo)) + (__inl13_ja[__sc0]))*(nk) + (__scs1)]));
              }
            }
          }
          int64_t *__inl13_jb = (int64_t *)malloc(((2)) * sizeof(int64_t));
          __inl13_jb[0] = nhalo;
          __inl13_jb[1] = (__inl13_j_end + 1);
          double *__inl13_left = (double *)malloc(((((2 * nhalo) + ni)) * (2) * (nk)) * sizeof(double));
          for (int64_t si0 = 0; si0 < ((2 * nhalo) + ni); ++si0) {
            for (int64_t si1 = 0; si1 < 2; ++si1) {
              for (int64_t si2 = 0; si2 < nk; ++si2) {
                __inl13_left[((si0)*(2) + (si1))*(nk) + (si2)] = (((((2.0 * dya[((si0)*(((nhalo + nj) + nhalo)) + ((__inl13_jb[si1] - 1)))*(nk) + (si2)]) + dya[((si0)*(((nhalo + nj) + nhalo)) + ((__inl13_jb[si1] - 2)))*(nk) + (si2)]) * q[((si0)*(((nhalo + nj) + nhalo)) + ((__inl13_jb[si1] - 1)))*(nk) + (si2)]) - (dya[((si0)*(((nhalo + nj) + nhalo)) + ((__inl13_jb[si1] - 1)))*(nk) + (si2)] * q[((si0)*(((nhalo + nj) + nhalo)) + ((__inl13_jb[si1] - 2)))*(nk) + (si2)])) / (dya[((si0)*(((nhalo + nj) + nhalo)) + ((__inl13_jb[si1] - 2)))*(nk) + (si2)] + dya[((si0)*(((nhalo + nj) + nhalo)) + ((__inl13_jb[si1] - 1)))*(nk) + (si2)]));
              }
            }
          }
          double *__inl13_right = (double *)malloc(((((2 * nhalo) + ni)) * (2) * (nk)) * sizeof(double));
          for (int64_t si0 = 0; si0 < ((2 * nhalo) + ni); ++si0) {
            for (int64_t si1 = 0; si1 < 2; ++si1) {
              for (int64_t si2 = 0; si2 < nk; ++si2) {
                __inl13_right[((si0)*(2) + (si1))*(nk) + (si2)] = (((((2.0 * dya[((si0)*(((nhalo + nj) + nhalo)) + (__inl13_jb[si1]))*(nk) + (si2)]) + dya[((si0)*(((nhalo + nj) + nhalo)) + ((__inl13_jb[si1] + 1)))*(nk) + (si2)]) * q[((si0)*(((nhalo + nj) + nhalo)) + (__inl13_jb[si1]))*(nk) + (si2)]) - (dya[((si0)*(((nhalo + nj) + nhalo)) + (__inl13_jb[si1]))*(nk) + (si2)] * q[((si0)*(((nhalo + nj) + nhalo)) + ((__inl13_jb[si1] + 1)))*(nk) + (si2)])) / (dya[((si0)*(((nhalo + nj) + nhalo)) + (__inl13_jb[si1]))*(nk) + (si2)] + dya[((si0)*(((nhalo + nj) + nhalo)) + ((__inl13_jb[si1] + 1)))*(nk) + (si2)]));
              }
            }
          }
          for (int64_t __scs0 = 0; __scs0 < __inl1_nx; ++__scs0) {
            for (int64_t __sc0 = 0; __sc0 < 2; ++__sc0) {
              for (int64_t __scs1 = 0; __scs1 < nk; ++__scs1) {
                __inl1_al[((__scs0)*(((2 * nhalo) + nj)) + (__inl13_jb[__sc0]))*(nk) + (__scs1)] = (0.5 * (__inl13_left[((__scs0)*(2) + (__sc0))*(nk) + (__scs1)] + __inl13_right[((__scs0)*(2) + (__sc0))*(nk) + (__scs1)]));
              }
            }
          }
          int64_t *__inl13_jc = (int64_t *)malloc(((2)) * sizeof(int64_t));
          __inl13_jc[0] = (nhalo + 1);
          __inl13_jc[1] = (__inl13_j_end + 2);
          for (int64_t __sc0 = 0; __sc0 < 2; ++__sc0) {
            for (int64_t __scs0 = 0; __scs0 < __inl1_nx; ++__scs0) {
              for (int64_t __scs1 = 0; __scs1 < nk; ++__scs1) {
                __inl1_al[((__scs0)*(((2 * nhalo) + nj)) + (__inl13_jc[__sc0]))*(nk) + (__scs1)] = (((0.35714285714285715 * q[((__scs0)*(((nhalo + nj) + nhalo)) + ((__inl13_jc[__sc0] - 1)))*(nk) + (__scs1)]) + (0.7857142857142857 * q[((__scs0)*(((nhalo + nj) + nhalo)) + (__inl13_jc[__sc0]))*(nk) + (__scs1)])) + (-0.14285714285714285 * q[((__scs0)*(((nhalo + nj) + nhalo)) + ((__inl13_jc[__sc0] + 1)))*(nk) + (__scs1)]));
              }
            }
          }
          free(__inl13_ja);
          free(__inl13_jb);
          free(__inl13_left);
          free(__inl13_right);
          free(__inl13_jc);
        }
        __inl14_j_end = ((nhalo + nj) - 1);
        __inl14_lo = nhalo;
        __inl14_hi = (__inl14_j_end + 2);
        for (int64_t __w0 = 0; __w0 < ((2 * nhalo) + ni); ++__w0) {
          for (int64_t __w1 = 0; __w1 < (__inl14_hi - nhalo); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl14_c[((__w0)*((__inl14_hi - nhalo)) + (__w1))*(nk) + (__w2)] = cry[((__w0)*(((nhalo + nj) + nhalo)) + ((__w1 + (nhalo - 0))))*(nk) + (__w2)];
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < ((2 * nhalo) + ni); ++__w0) {
          for (int64_t __w1 = 0; __w1 < (__inl14_hi - nhalo); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl14_q_j[((__w0)*((__inl14_hi - nhalo)) + (__w1))*(nk) + (__w2)] = q[((__w0)*(((nhalo + nj) + nhalo)) + ((__w1 + (nhalo - 0))))*(nk) + (__w2)];
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < ((2 * nhalo) + ni); ++__w0) {
          for (int64_t __w1 = 0; __w1 < ((__inl14_hi - 1) - (nhalo - 1)); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl14_q_jm1[((__w0)*(((__inl14_hi - 1) - (nhalo - 1))) + (__w1))*(nk) + (__w2)] = q[((__w0)*(((nhalo + nj) + nhalo)) + ((__w1 + ((nhalo - 1) - 0))))*(nk) + (__w2)];
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < __inl1_nx; ++__w0) {
          for (int64_t __w1 = 0; __w1 < (__inl14_hi - nhalo); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl14_bl[((__w0)*((__inl14_hi - nhalo)) + (__w1))*(nk) + (__w2)] = (__inl1_al[((__w0)*(((2 * nhalo) + nj)) + ((__w1 + (nhalo - 0))))*(nk) + (__w2)] - __inl14_q_j[((__w0)*((__inl14_hi - nhalo)) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < __inl1_nx; ++__w0) {
          for (int64_t __w1 = 0; __w1 < ((__inl14_hi + 1) - (nhalo + 1)); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl14_br[((__w0)*(((__inl14_hi + 1) - (nhalo + 1))) + (__w1))*(nk) + (__w2)] = (__inl1_al[((__w0)*(((2 * nhalo) + nj)) + ((__w1 + ((nhalo + 1) - 0))))*(nk) + (__w2)] - __inl14_q_j[((__w0)*((__inl14_hi - nhalo)) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < __inl1_nx; ++__w0) {
          for (int64_t __w1 = 0; __w1 < (__inl14_hi - nhalo); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl14_b0[((__w0)*((__inl14_hi - nhalo)) + (__w1))*(nk) + (__w2)] = (__inl14_bl[((__w0)*((__inl14_hi - nhalo)) + (__w1))*(nk) + (__w2)] + __inl14_br[((__w0)*(((__inl14_hi + 1) - (nhalo + 1))) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < __inl1_nx; ++__w0) {
          for (int64_t __w1 = 0; __w1 < ((__inl14_hi - 1) - (nhalo - 1)); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl14_bl_m1[((__w0)*(((__inl14_hi - 1) - (nhalo - 1))) + (__w1))*(nk) + (__w2)] = (__inl1_al[((__w0)*(((2 * nhalo) + nj)) + ((__w1 + ((nhalo - 1) - 0))))*(nk) + (__w2)] - __inl14_q_jm1[((__w0)*(((__inl14_hi - 1) - (nhalo - 1))) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < __inl1_nx; ++__w0) {
          for (int64_t __w1 = 0; __w1 < (__inl14_hi - nhalo); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl14_br_m1[((__w0)*((__inl14_hi - nhalo)) + (__w1))*(nk) + (__w2)] = (__inl1_al[((__w0)*(((2 * nhalo) + nj)) + ((__w1 + (nhalo - 0))))*(nk) + (__w2)] - __inl14_q_jm1[((__w0)*(((__inl14_hi - 1) - (nhalo - 1))) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < __inl1_nx; ++__w0) {
          for (int64_t __w1 = 0; __w1 < ((__inl14_hi - 1) - (nhalo - 1)); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl14_b0_m1[((__w0)*(((__inl14_hi - 1) - (nhalo - 1))) + (__w1))*(nk) + (__w2)] = (__inl14_bl_m1[((__w0)*(((__inl14_hi - 1) - (nhalo - 1))) + (__w1))*(nk) + (__w2)] + __inl14_br_m1[((__w0)*((__inl14_hi - nhalo)) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        if ((llabs(__inl1_ord_inner) == 5)) {
          for (int64_t __w0 = 0; __w0 < __inl1_nx; ++__w0) {
            for (int64_t __w1 = 0; __w1 < (__inl14_hi - nhalo); ++__w1) {
              for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
                __inl14_smt5[((__w0)*((__inl14_hi - nhalo)) + (__w1))*(nk) + (__w2)] = ((__inl14_bl[((__w0)*((__inl14_hi - nhalo)) + (__w1))*(nk) + (__w2)] * __inl14_br[((__w0)*(((__inl14_hi + 1) - (nhalo + 1))) + (__w1))*(nk) + (__w2)]) < 0.0);
              }
            }
          }
          for (int64_t __w0 = 0; __w0 < __inl1_nx; ++__w0) {
            for (int64_t __w1 = 0; __w1 < ((__inl14_hi - 1) - (nhalo - 1)); ++__w1) {
              for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
                __inl14_smt5_m1[((__w0)*(((__inl14_hi - 1) - (nhalo - 1))) + (__w1))*(nk) + (__w2)] = ((__inl14_bl_m1[((__w0)*(((__inl14_hi - 1) - (nhalo - 1))) + (__w1))*(nk) + (__w2)] * __inl14_br_m1[((__w0)*((__inl14_hi - nhalo)) + (__w1))*(nk) + (__w2)]) < 0.0);
              }
            }
          }
        }
        else {
          for (int64_t __w0 = 0; __w0 < __inl1_nx; ++__w0) {
            for (int64_t __w1 = 0; __w1 < (__inl14_hi - nhalo); ++__w1) {
              for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
                __inl14_smt5[((__w0)*((__inl14_hi - nhalo)) + (__w1))*(nk) + (__w2)] = ((3.0 * fabs(__inl14_b0[((__w0)*((__inl14_hi - nhalo)) + (__w1))*(nk) + (__w2)])) < fabs((__inl14_bl[((__w0)*((__inl14_hi - nhalo)) + (__w1))*(nk) + (__w2)] - __inl14_br[((__w0)*(((__inl14_hi + 1) - (nhalo + 1))) + (__w1))*(nk) + (__w2)])));
              }
            }
          }
          for (int64_t __w0 = 0; __w0 < __inl1_nx; ++__w0) {
            for (int64_t __w1 = 0; __w1 < ((__inl14_hi - 1) - (nhalo - 1)); ++__w1) {
              for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
                __inl14_smt5_m1[((__w0)*(((__inl14_hi - 1) - (nhalo - 1))) + (__w1))*(nk) + (__w2)] = ((3.0 * fabs(__inl14_b0_m1[((__w0)*(((__inl14_hi - 1) - (nhalo - 1))) + (__w1))*(nk) + (__w2)])) < fabs((__inl14_bl_m1[((__w0)*(((__inl14_hi - 1) - (nhalo - 1))) + (__w1))*(nk) + (__w2)] - __inl14_br_m1[((__w0)*((__inl14_hi - nhalo)) + (__w1))*(nk) + (__w2)])));
              }
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < __inl1_nx; ++__w0) {
          for (int64_t __w1 = 0; __w1 < (__inl14_hi - nhalo); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl14_mask[((__w0)*((__inl14_hi - nhalo)) + (__w1))*(nk) + (__w2)] = ((double)((__inl14_smt5[((__w0)*((__inl14_hi - nhalo)) + (__w1))*(nk) + (__w2)] | __inl14_smt5_m1[((__w0)*(((__inl14_hi - 1) - (nhalo - 1))) + (__w1))*(nk) + (__w2)])));
            }
          }
        }
        /* numpy: np.where(__inl14_c > 0.0, __inl14_q_jm1 + (1.0 - __inl14_c) * (__inl14_br_m1 - __inl14_c * __... */
        for (int64_t __r0 = 0; __r0 < ((2 * nhalo) + ni); ++__r0) {
          for (int64_t __r1 = 0; __r1 < (__inl14_hi - nhalo); ++__r1) {
            for (int64_t __r2 = 0; __r2 < nk; ++__r2) {
              __cb1[((__r0)*((nj + 1)) + (__r1))*(nk) + (__r2)] = ((__inl14_c[((__r0)*((__inl14_hi - nhalo)) + (__r1))*(nk) + (__r2)] > 0.0) ? (__inl14_q_jm1[((__r0)*(((__inl14_hi - 1) - (nhalo - 1))) + (__r1))*(nk) + (__r2)] + (((1.0 - __inl14_c[((__r0)*((__inl14_hi - nhalo)) + (__r1))*(nk) + (__r2)]) * (__inl14_br_m1[((__r0)*((__inl14_hi - nhalo)) + (__r1))*(nk) + (__r2)] - (__inl14_c[((__r0)*((__inl14_hi - nhalo)) + (__r1))*(nk) + (__r2)] * __inl14_b0_m1[((__r0)*(((__inl14_hi - 1) - (nhalo - 1))) + (__r1))*(nk) + (__r2)]))) * __inl14_mask[((__r0)*((__inl14_hi - nhalo)) + (__r1))*(nk) + (__r2)])) : (__inl14_q_j[((__r0)*((__inl14_hi - nhalo)) + (__r1))*(nk) + (__r2)] + (((1.0 + __inl14_c[((__r0)*((__inl14_hi - nhalo)) + (__r1))*(nk) + (__r2)]) * (__inl14_bl[((__r0)*((__inl14_hi - nhalo)) + (__r1))*(nk) + (__r2)] + (__inl14_c[((__r0)*((__inl14_hi - nhalo)) + (__r1))*(nk) + (__r2)] * __inl14_b0[((__r0)*((__inl14_hi - nhalo)) + (__r1))*(nk) + (__r2)]))) * __inl14_mask[((__r0)*((__inl14_hi - nhalo)) + (__r1))*(nk) + (__r2)])));
            }
          }
        }
        for (int64_t si0 = 0; si0 < ((2 * nhalo) + ni); ++si0) {
          for (int64_t si1 = __inl14_lo; si1 < __inl14_hi; ++si1) {
            for (int64_t si2 = 0; si2 < nk; ++si2) {
              __inl1_q_y_advected_mean[((si0)*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)] = __cb1[((si0)*((nj + 1)) + ((si1 - __inl14_lo)))*(nk) + (si2)];
            }
          }
        }
        __inl4_ny = ((nhalo + nj) + nhalo);
        __inl4_j0 = 3;
        __inl4_j1 = (__inl4_ny - 3);
        for (int64_t __w0 = 0; __w0 < ((2 * nhalo) + ni); ++__w0) {
          for (int64_t __w1 = 0; __w1 < (__inl4_j1 - __inl4_j0); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl4_fyy_j[((__w0)*((__inl4_j1 - __inl4_j0)) + (__w1))*(nk) + (__w2)] = (y_area_flux[((__w0)*(((nhalo + nj) + nhalo)) + ((__w1 + (__inl4_j0 - 0))))*(nk) + (__w2)] * __inl1_q_y_advected_mean[((__w0)*(((2 * nhalo) + nj)) + ((__w1 + (__inl4_j0 - 0))))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < ((2 * nhalo) + ni); ++__w0) {
          for (int64_t __w1 = 0; __w1 < ((__inl4_j1 + 1) - (__inl4_j0 + 1)); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl4_fyy_jp1[((__w0)*(((__inl4_j1 + 1) - (__inl4_j0 + 1))) + (__w1))*(nk) + (__w2)] = (y_area_flux[((__w0)*(((nhalo + nj) + nhalo)) + ((__w1 + ((__inl4_j0 + 1) - 0))))*(nk) + (__w2)] * __inl1_q_y_advected_mean[((__w0)*(((2 * nhalo) + nj)) + ((__w1 + ((__inl4_j0 + 1) - 0))))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < ((2 * nhalo) + ni); ++__w0) {
          for (int64_t __w1 = 0; __w1 < (__inl4_j1 - __inl4_j0); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl4_denom[((__w0)*((__inl4_j1 - __inl4_j0)) + (__w1))*(nk) + (__w2)] = ((area[((__w0)*(((nhalo + nj) + nhalo)) + ((__w1 + (__inl4_j0 - 0))))*(nk) + (__w2)] + y_area_flux[((__w0)*(((nhalo + nj) + nhalo)) + ((__w1 + (__inl4_j0 - 0))))*(nk) + (__w2)]) - y_area_flux[((__w0)*(((nhalo + nj) + nhalo)) + ((__w1 + ((__inl4_j0 + 1) - 0))))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t si0 = 0; si0 < ((2 * nhalo) + ni); ++si0) {
          for (int64_t si1 = __inl4_j0; si1 < __inl4_j1; ++si1) {
            for (int64_t si2 = 0; si2 < nk; ++si2) {
              __inl1_q_advected_y[((si0)*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)] = ((((q[((si0)*(((nhalo + nj) + nhalo)) + ((si1 + (__inl4_j0 - __inl4_j0))))*(nk) + (si2)] * area[((si0)*(((nhalo + nj) + nhalo)) + ((si1 + (__inl4_j0 - __inl4_j0))))*(nk) + (si2)]) + __inl4_fyy_j[((si0)*((__inl4_j1 - __inl4_j0)) + ((si1 - __inl4_j0)))*(nk) + (si2)]) - __inl4_fyy_jp1[((si0)*(((__inl4_j1 + 1) - (__inl4_j0 + 1))) + ((si1 - __inl4_j0)))*(nk) + (si2)]) / __inl4_denom[((si0)*((__inl4_j1 - __inl4_j0)) + ((si1 - __inl4_j0)))*(nk) + (si2)]);
            }
          }
        }
        __inl15_i_end = ((nhalo + ni) - 1);
        __inl15_lo = (nhalo - 1);
        __inl15_hi = (__inl15_i_end + 3);
        for (int64_t si0 = __inl15_lo; si0 < __inl15_hi; ++si0) {
          for (int64_t si1 = 0; si1 < ((2 * nhalo) + nj); ++si1) {
            for (int64_t si2 = 0; si2 < nk; ++si2) {
              __inl1_al[((si0)*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)] = ((0.5833333333333334 * (__inl1_q_advected_y[(((si0 + ((__inl15_lo - 1) - __inl15_lo)))*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)] + __inl1_q_advected_y[(((si0 + (__inl15_lo - __inl15_lo)))*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)])) + (-0.08333333333333333 * (__inl1_q_advected_y[(((si0 + ((__inl15_lo - 2) - __inl15_lo)))*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)] + __inl1_q_advected_y[(((si0 + ((__inl15_lo + 1) - __inl15_lo)))*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)])));
            }
          }
        }
        if ((grid_type < 3)) {
          int64_t *__inl15_ia = (int64_t *)malloc(((2)) * sizeof(int64_t));
          __inl15_ia[0] = (nhalo - 1);
          __inl15_ia[1] = __inl15_i_end;
          for (int64_t __sc0 = 0; __sc0 < 2; ++__sc0) {
            for (int64_t __scs0 = 0; __scs0 < __inl1_ny; ++__scs0) {
              for (int64_t __scs1 = 0; __scs1 < nk; ++__scs1) {
                __inl1_al[((__inl15_ia[__sc0])*(((2 * nhalo) + nj)) + (__scs0))*(nk) + (__scs1)] = (((-0.14285714285714285 * __inl1_q_advected_y[(((__inl15_ia[__sc0] - 2))*(((2 * nhalo) + nj)) + (__scs0))*(nk) + (__scs1)]) + (0.7857142857142857 * __inl1_q_advected_y[(((__inl15_ia[__sc0] - 1))*(((2 * nhalo) + nj)) + (__scs0))*(nk) + (__scs1)])) + (0.35714285714285715 * __inl1_q_advected_y[((__inl15_ia[__sc0])*(((2 * nhalo) + nj)) + (__scs0))*(nk) + (__scs1)]));
              }
            }
          }
          int64_t *__inl15_ib = (int64_t *)malloc(((2)) * sizeof(int64_t));
          __inl15_ib[0] = nhalo;
          __inl15_ib[1] = (__inl15_i_end + 1);
          double *__inl15_left = (double *)malloc(((2) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
          for (int64_t si0 = 0; si0 < 2; ++si0) {
            for (int64_t si1 = 0; si1 < ((2 * nhalo) + nj); ++si1) {
              for (int64_t si2 = 0; si2 < nk; ++si2) {
                __inl15_left[((si0)*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)] = (((((2.0 * dxa[(((__inl15_ib[si0] - 1))*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)]) + dxa[(((__inl15_ib[si0] - 2))*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)]) * __inl1_q_advected_y[(((__inl15_ib[si0] - 1))*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)]) - (dxa[(((__inl15_ib[si0] - 1))*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)] * __inl1_q_advected_y[(((__inl15_ib[si0] - 2))*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)])) / (dxa[(((__inl15_ib[si0] - 2))*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)] + dxa[(((__inl15_ib[si0] - 1))*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)]));
              }
            }
          }
          double *__inl15_right = (double *)malloc(((2) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
          for (int64_t si0 = 0; si0 < 2; ++si0) {
            for (int64_t si1 = 0; si1 < ((2 * nhalo) + nj); ++si1) {
              for (int64_t si2 = 0; si2 < nk; ++si2) {
                __inl15_right[((si0)*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)] = (((((2.0 * dxa[((__inl15_ib[si0])*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)]) + dxa[(((__inl15_ib[si0] + 1))*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)]) * __inl1_q_advected_y[((__inl15_ib[si0])*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)]) - (dxa[((__inl15_ib[si0])*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)] * __inl1_q_advected_y[(((__inl15_ib[si0] + 1))*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)])) / (dxa[((__inl15_ib[si0])*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)] + dxa[(((__inl15_ib[si0] + 1))*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)]));
              }
            }
          }
          for (int64_t __sc0 = 0; __sc0 < 2; ++__sc0) {
            for (int64_t __scs0 = 0; __scs0 < __inl1_ny; ++__scs0) {
              for (int64_t __scs1 = 0; __scs1 < nk; ++__scs1) {
                __inl1_al[((__inl15_ib[__sc0])*(((2 * nhalo) + nj)) + (__scs0))*(nk) + (__scs1)] = (0.5 * (__inl15_left[((__sc0)*(((2 * nhalo) + nj)) + (__scs0))*(nk) + (__scs1)] + __inl15_right[((__sc0)*(((2 * nhalo) + nj)) + (__scs0))*(nk) + (__scs1)]));
              }
            }
          }
          int64_t *__inl15_ic = (int64_t *)malloc(((2)) * sizeof(int64_t));
          __inl15_ic[0] = (nhalo + 1);
          __inl15_ic[1] = (__inl15_i_end + 2);
          for (int64_t __sc0 = 0; __sc0 < 2; ++__sc0) {
            for (int64_t __scs0 = 0; __scs0 < __inl1_ny; ++__scs0) {
              for (int64_t __scs1 = 0; __scs1 < nk; ++__scs1) {
                __inl1_al[((__inl15_ic[__sc0])*(((2 * nhalo) + nj)) + (__scs0))*(nk) + (__scs1)] = (((0.35714285714285715 * __inl1_q_advected_y[(((__inl15_ic[__sc0] - 1))*(((2 * nhalo) + nj)) + (__scs0))*(nk) + (__scs1)]) + (0.7857142857142857 * __inl1_q_advected_y[((__inl15_ic[__sc0])*(((2 * nhalo) + nj)) + (__scs0))*(nk) + (__scs1)])) + (-0.14285714285714285 * __inl1_q_advected_y[(((__inl15_ic[__sc0] + 1))*(((2 * nhalo) + nj)) + (__scs0))*(nk) + (__scs1)]));
              }
            }
          }
          free(__inl15_ia);
          free(__inl15_ib);
          free(__inl15_left);
          free(__inl15_right);
          free(__inl15_ic);
        }
        __inl16_i_end = ((nhalo + ni) - 1);
        __inl16_lo = nhalo;
        __inl16_hi = (__inl16_i_end + 2);
        for (int64_t __w0 = 0; __w0 < (__inl16_hi - nhalo); ++__w0) {
          for (int64_t __w1 = 0; __w1 < ((2 * nhalo) + nj); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl16_c[((__w0)*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)] = crx[(((__w0 + (nhalo - 0)))*(((nhalo + nj) + nhalo)) + (__w1))*(nk) + (__w2)];
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < (__inl16_hi - nhalo); ++__w0) {
          for (int64_t __w1 = 0; __w1 < ((2 * nhalo) + nj); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl16_q_i[((__w0)*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)] = __inl1_q_advected_y[(((__w0 + (nhalo - 0)))*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)];
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < ((__inl16_hi - 1) - (nhalo - 1)); ++__w0) {
          for (int64_t __w1 = 0; __w1 < ((2 * nhalo) + nj); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl16_q_im1[((__w0)*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)] = __inl1_q_advected_y[(((__w0 + ((nhalo - 1) - 0)))*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)];
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < (__inl16_hi - nhalo); ++__w0) {
          for (int64_t __w1 = 0; __w1 < __inl1_ny; ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl16_bl[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] = (__inl1_al[(((__w0 + (nhalo - 0)))*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)] - __inl16_q_i[((__w0)*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < ((__inl16_hi + 1) - (nhalo + 1)); ++__w0) {
          for (int64_t __w1 = 0; __w1 < __inl1_ny; ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl16_br[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] = (__inl1_al[(((__w0 + ((nhalo + 1) - 0)))*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)] - __inl16_q_i[((__w0)*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < (__inl16_hi - nhalo); ++__w0) {
          for (int64_t __w1 = 0; __w1 < __inl1_ny; ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl16_b0[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] = (__inl16_bl[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] + __inl16_br[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < ((__inl16_hi - 1) - (nhalo - 1)); ++__w0) {
          for (int64_t __w1 = 0; __w1 < __inl1_ny; ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl16_bl_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] = (__inl1_al[(((__w0 + ((nhalo - 1) - 0)))*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)] - __inl16_q_im1[((__w0)*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < (__inl16_hi - nhalo); ++__w0) {
          for (int64_t __w1 = 0; __w1 < __inl1_ny; ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl16_br_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] = (__inl1_al[(((__w0 + (nhalo - 0)))*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)] - __inl16_q_im1[((__w0)*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < ((__inl16_hi - 1) - (nhalo - 1)); ++__w0) {
          for (int64_t __w1 = 0; __w1 < __inl1_ny; ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl16_b0_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] = (__inl16_bl_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] + __inl16_br_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        if ((llabs(hord) == 5)) {
          for (int64_t __w0 = 0; __w0 < (__inl16_hi - nhalo); ++__w0) {
            for (int64_t __w1 = 0; __w1 < __inl1_ny; ++__w1) {
              for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
                __inl16_smt5[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] = ((__inl16_bl[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] * __inl16_br[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)]) < 0.0);
              }
            }
          }
          for (int64_t __w0 = 0; __w0 < ((__inl16_hi - 1) - (nhalo - 1)); ++__w0) {
            for (int64_t __w1 = 0; __w1 < __inl1_ny; ++__w1) {
              for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
                __inl16_smt5_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] = ((__inl16_bl_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] * __inl16_br_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)]) < 0.0);
              }
            }
          }
        }
        else {
          for (int64_t __w0 = 0; __w0 < (__inl16_hi - nhalo); ++__w0) {
            for (int64_t __w1 = 0; __w1 < __inl1_ny; ++__w1) {
              for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
                __inl16_smt5[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] = ((3.0 * fabs(__inl16_b0[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)])) < fabs((__inl16_bl[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] - __inl16_br[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)])));
              }
            }
          }
          for (int64_t __w0 = 0; __w0 < ((__inl16_hi - 1) - (nhalo - 1)); ++__w0) {
            for (int64_t __w1 = 0; __w1 < __inl1_ny; ++__w1) {
              for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
                __inl16_smt5_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] = ((3.0 * fabs(__inl16_b0_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)])) < fabs((__inl16_bl_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] - __inl16_br_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)])));
              }
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < (__inl16_hi - nhalo); ++__w0) {
          for (int64_t __w1 = 0; __w1 < __inl1_ny; ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl16_mask[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] = ((double)((__inl16_smt5[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] | __inl16_smt5_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)])));
            }
          }
        }
        /* numpy: np.where(__inl16_c > 0.0, __inl16_q_im1 + (1.0 - __inl16_c) * (__inl16_br_m1 - __inl16_c * __... */
        for (int64_t __r0 = 0; __r0 < (__inl16_hi - nhalo); ++__r0) {
          for (int64_t __r1 = 0; __r1 < ((2 * nhalo) + nj); ++__r1) {
            for (int64_t __r2 = 0; __r2 < nk; ++__r2) {
              __cb2[((__r0)*(((2 * nhalo) + nj)) + (__r1))*(nk) + (__r2)] = ((__inl16_c[((__r0)*(((2 * nhalo) + nj)) + (__r1))*(nk) + (__r2)] > 0.0) ? (__inl16_q_im1[((__r0)*(((2 * nhalo) + nj)) + (__r1))*(nk) + (__r2)] + (((1.0 - __inl16_c[((__r0)*(((2 * nhalo) + nj)) + (__r1))*(nk) + (__r2)]) * (__inl16_br_m1[((__r0)*(__inl1_ny) + (__r1))*(nk) + (__r2)] - (__inl16_c[((__r0)*(((2 * nhalo) + nj)) + (__r1))*(nk) + (__r2)] * __inl16_b0_m1[((__r0)*(__inl1_ny) + (__r1))*(nk) + (__r2)]))) * __inl16_mask[((__r0)*(__inl1_ny) + (__r1))*(nk) + (__r2)])) : (__inl16_q_i[((__r0)*(((2 * nhalo) + nj)) + (__r1))*(nk) + (__r2)] + (((1.0 + __inl16_c[((__r0)*(((2 * nhalo) + nj)) + (__r1))*(nk) + (__r2)]) * (__inl16_bl[((__r0)*(__inl1_ny) + (__r1))*(nk) + (__r2)] + (__inl16_c[((__r0)*(((2 * nhalo) + nj)) + (__r1))*(nk) + (__r2)] * __inl16_b0[((__r0)*(__inl1_ny) + (__r1))*(nk) + (__r2)]))) * __inl16_mask[((__r0)*(__inl1_ny) + (__r1))*(nk) + (__r2)])));
            }
          }
        }
        for (int64_t si0 = __inl16_lo; si0 < __inl16_hi; ++si0) {
          for (int64_t si1 = 0; si1 < ((2 * nhalo) + nj); ++si1) {
            for (int64_t si2 = 0; si2 < nk; ++si2) {
              __inl1_q_ayxa[((si0)*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)] = __cb2[(((si0 - __inl16_lo))*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)];
            }
          }
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((0)*(((nhalo + nj) + nhalo)) + (0))*(nk) + (si2)] = q[((0)*(((nhalo + nj) + nhalo)) + (5))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((0)*(((nhalo + nj) + nhalo)) + (1))*(nk) + (si2)] = q[((1)*(((nhalo + nj) + nhalo)) + (5))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((0)*(((nhalo + nj) + nhalo)) + (2))*(nk) + (si2)] = q[((2)*(((nhalo + nj) + nhalo)) + (5))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((1)*(((nhalo + nj) + nhalo)) + (0))*(nk) + (si2)] = q[((0)*(((nhalo + nj) + nhalo)) + (4))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((1)*(((nhalo + nj) + nhalo)) + (1))*(nk) + (si2)] = q[((1)*(((nhalo + nj) + nhalo)) + (4))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((1)*(((nhalo + nj) + nhalo)) + (2))*(nk) + (si2)] = q[((2)*(((nhalo + nj) + nhalo)) + (4))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((2)*(((nhalo + nj) + nhalo)) + (0))*(nk) + (si2)] = q[((0)*(((nhalo + nj) + nhalo)) + (3))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((2)*(((nhalo + nj) + nhalo)) + (1))*(nk) + (si2)] = q[((1)*(((nhalo + nj) + nhalo)) + (3))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((2)*(((nhalo + nj) + nhalo)) + (2))*(nk) + (si2)] = q[((2)*(((nhalo + nj) + nhalo)) + (3))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((0)*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 4)))*(nk) + (si2)] = q[((2)*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 7)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((0)*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 3)))*(nk) + (si2)] = q[((1)*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 7)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((0)*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 2)))*(nk) + (si2)] = q[((0)*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 7)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((1)*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 4)))*(nk) + (si2)] = q[((2)*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 6)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((1)*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 3)))*(nk) + (si2)] = q[((1)*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 6)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((1)*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 2)))*(nk) + (si2)] = q[((0)*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 6)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((2)*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 4)))*(nk) + (si2)] = q[((2)*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 5)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((2)*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 3)))*(nk) + (si2)] = q[((1)*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 5)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[((2)*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 2)))*(nk) + (si2)] = q[((0)*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 5)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 4))*(((nhalo + nj) + nhalo)) + (0))*(nk) + (si2)] = q[(((2 * nhalo + ni - 2))*(((nhalo + nj) + nhalo)) + (3))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 4))*(((nhalo + nj) + nhalo)) + (1))*(nk) + (si2)] = q[(((2 * nhalo + ni - 3))*(((nhalo + nj) + nhalo)) + (3))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 4))*(((nhalo + nj) + nhalo)) + (2))*(nk) + (si2)] = q[(((2 * nhalo + ni - 4))*(((nhalo + nj) + nhalo)) + (3))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 3))*(((nhalo + nj) + nhalo)) + (0))*(nk) + (si2)] = q[(((2 * nhalo + ni - 2))*(((nhalo + nj) + nhalo)) + (4))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 3))*(((nhalo + nj) + nhalo)) + (1))*(nk) + (si2)] = q[(((2 * nhalo + ni - 3))*(((nhalo + nj) + nhalo)) + (4))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 3))*(((nhalo + nj) + nhalo)) + (2))*(nk) + (si2)] = q[(((2 * nhalo + ni - 4))*(((nhalo + nj) + nhalo)) + (4))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 2))*(((nhalo + nj) + nhalo)) + (0))*(nk) + (si2)] = q[(((2 * nhalo + ni - 2))*(((nhalo + nj) + nhalo)) + (5))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 2))*(((nhalo + nj) + nhalo)) + (1))*(nk) + (si2)] = q[(((2 * nhalo + ni - 3))*(((nhalo + nj) + nhalo)) + (5))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 2))*(((nhalo + nj) + nhalo)) + (2))*(nk) + (si2)] = q[(((2 * nhalo + ni - 4))*(((nhalo + nj) + nhalo)) + (5))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 4))*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 2)))*(nk) + (si2)] = q[(((2 * nhalo + ni - 2))*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 5)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 4))*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 3)))*(nk) + (si2)] = q[(((2 * nhalo + ni - 3))*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 5)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 4))*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 4)))*(nk) + (si2)] = q[(((2 * nhalo + ni - 4))*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 5)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 3))*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 2)))*(nk) + (si2)] = q[(((2 * nhalo + ni - 2))*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 6)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 3))*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 3)))*(nk) + (si2)] = q[(((2 * nhalo + ni - 3))*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 6)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 3))*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 4)))*(nk) + (si2)] = q[(((2 * nhalo + ni - 4))*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 6)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 2))*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 2)))*(nk) + (si2)] = q[(((2 * nhalo + ni - 2))*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 7)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 2))*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 3)))*(nk) + (si2)] = q[(((2 * nhalo + ni - 3))*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 7)))*(nk) + (si2)];
        }
        for (int64_t si2 = 0; si2 < nk; ++si2) {
          q[(((((2 * nhalo) + ni) - 2))*(((nhalo + nj) + nhalo)) + ((((2 * nhalo) + nj) - 4)))*(nk) + (si2)] = q[(((2 * nhalo + ni - 4))*(((nhalo + nj) + nhalo)) + ((2 * nhalo + nj - 7)))*(nk) + (si2)];
        }
        __inl17_i_end = ((nhalo + ni) - 1);
        __inl17_lo = (nhalo - 1);
        __inl17_hi = (__inl17_i_end + 3);
        for (int64_t si0 = __inl17_lo; si0 < __inl17_hi; ++si0) {
          for (int64_t si1 = 0; si1 < ((2 * nhalo) + nj); ++si1) {
            for (int64_t si2 = 0; si2 < nk; ++si2) {
              __inl1_al[((si0)*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)] = ((0.5833333333333334 * (q[(((si0 + ((__inl17_lo - 1) - __inl17_lo)))*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)] + q[(((si0 + (__inl17_lo - __inl17_lo)))*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)])) + (-0.08333333333333333 * (q[(((si0 + ((__inl17_lo - 2) - __inl17_lo)))*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)] + q[(((si0 + ((__inl17_lo + 1) - __inl17_lo)))*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)])));
            }
          }
        }
        if ((grid_type < 3)) {
          int64_t *__inl17_ia = (int64_t *)malloc(((2)) * sizeof(int64_t));
          __inl17_ia[0] = (nhalo - 1);
          __inl17_ia[1] = __inl17_i_end;
          for (int64_t __sc0 = 0; __sc0 < 2; ++__sc0) {
            for (int64_t __scs0 = 0; __scs0 < __inl1_ny; ++__scs0) {
              for (int64_t __scs1 = 0; __scs1 < nk; ++__scs1) {
                __inl1_al[((__inl17_ia[__sc0])*(((2 * nhalo) + nj)) + (__scs0))*(nk) + (__scs1)] = (((-0.14285714285714285 * q[(((__inl17_ia[__sc0] - 2))*(((nhalo + nj) + nhalo)) + (__scs0))*(nk) + (__scs1)]) + (0.7857142857142857 * q[(((__inl17_ia[__sc0] - 1))*(((nhalo + nj) + nhalo)) + (__scs0))*(nk) + (__scs1)])) + (0.35714285714285715 * q[((__inl17_ia[__sc0])*(((nhalo + nj) + nhalo)) + (__scs0))*(nk) + (__scs1)]));
              }
            }
          }
          int64_t *__inl17_ib = (int64_t *)malloc(((2)) * sizeof(int64_t));
          __inl17_ib[0] = nhalo;
          __inl17_ib[1] = (__inl17_i_end + 1);
          double *__inl17_left = (double *)malloc(((2) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
          for (int64_t si0 = 0; si0 < 2; ++si0) {
            for (int64_t si1 = 0; si1 < ((2 * nhalo) + nj); ++si1) {
              for (int64_t si2 = 0; si2 < nk; ++si2) {
                __inl17_left[((si0)*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)] = (((((2.0 * dxa[(((__inl17_ib[si0] - 1))*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)]) + dxa[(((__inl17_ib[si0] - 2))*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)]) * q[(((__inl17_ib[si0] - 1))*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)]) - (dxa[(((__inl17_ib[si0] - 1))*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)] * q[(((__inl17_ib[si0] - 2))*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)])) / (dxa[(((__inl17_ib[si0] - 2))*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)] + dxa[(((__inl17_ib[si0] - 1))*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)]));
              }
            }
          }
          double *__inl17_right = (double *)malloc(((2) * (((2 * nhalo) + nj)) * (nk)) * sizeof(double));
          for (int64_t si0 = 0; si0 < 2; ++si0) {
            for (int64_t si1 = 0; si1 < ((2 * nhalo) + nj); ++si1) {
              for (int64_t si2 = 0; si2 < nk; ++si2) {
                __inl17_right[((si0)*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)] = (((((2.0 * dxa[((__inl17_ib[si0])*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)]) + dxa[(((__inl17_ib[si0] + 1))*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)]) * q[((__inl17_ib[si0])*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)]) - (dxa[((__inl17_ib[si0])*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)] * q[(((__inl17_ib[si0] + 1))*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)])) / (dxa[((__inl17_ib[si0])*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)] + dxa[(((__inl17_ib[si0] + 1))*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)]));
              }
            }
          }
          for (int64_t __sc0 = 0; __sc0 < 2; ++__sc0) {
            for (int64_t __scs0 = 0; __scs0 < __inl1_ny; ++__scs0) {
              for (int64_t __scs1 = 0; __scs1 < nk; ++__scs1) {
                __inl1_al[((__inl17_ib[__sc0])*(((2 * nhalo) + nj)) + (__scs0))*(nk) + (__scs1)] = (0.5 * (__inl17_left[((__sc0)*(((2 * nhalo) + nj)) + (__scs0))*(nk) + (__scs1)] + __inl17_right[((__sc0)*(((2 * nhalo) + nj)) + (__scs0))*(nk) + (__scs1)]));
              }
            }
          }
          int64_t *__inl17_ic = (int64_t *)malloc(((2)) * sizeof(int64_t));
          __inl17_ic[0] = (nhalo + 1);
          __inl17_ic[1] = (__inl17_i_end + 2);
          for (int64_t __sc0 = 0; __sc0 < 2; ++__sc0) {
            for (int64_t __scs0 = 0; __scs0 < __inl1_ny; ++__scs0) {
              for (int64_t __scs1 = 0; __scs1 < nk; ++__scs1) {
                __inl1_al[((__inl17_ic[__sc0])*(((2 * nhalo) + nj)) + (__scs0))*(nk) + (__scs1)] = (((0.35714285714285715 * q[(((__inl17_ic[__sc0] - 1))*(((nhalo + nj) + nhalo)) + (__scs0))*(nk) + (__scs1)]) + (0.7857142857142857 * q[((__inl17_ic[__sc0])*(((nhalo + nj) + nhalo)) + (__scs0))*(nk) + (__scs1)])) + (-0.14285714285714285 * q[(((__inl17_ic[__sc0] + 1))*(((nhalo + nj) + nhalo)) + (__scs0))*(nk) + (__scs1)]));
              }
            }
          }
          free(__inl17_ia);
          free(__inl17_ib);
          free(__inl17_left);
          free(__inl17_right);
          free(__inl17_ic);
        }
        __inl18_i_end = ((nhalo + ni) - 1);
        __inl18_lo = nhalo;
        __inl18_hi = (__inl18_i_end + 2);
        for (int64_t __w0 = 0; __w0 < (__inl18_hi - nhalo); ++__w0) {
          for (int64_t __w1 = 0; __w1 < ((2 * nhalo) + nj); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl18_c[((__w0)*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)] = crx[(((__w0 + (nhalo - 0)))*(((nhalo + nj) + nhalo)) + (__w1))*(nk) + (__w2)];
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < (__inl18_hi - nhalo); ++__w0) {
          for (int64_t __w1 = 0; __w1 < ((2 * nhalo) + nj); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl18_q_i[((__w0)*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)] = q[(((__w0 + (nhalo - 0)))*(((nhalo + nj) + nhalo)) + (__w1))*(nk) + (__w2)];
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < ((__inl18_hi - 1) - (nhalo - 1)); ++__w0) {
          for (int64_t __w1 = 0; __w1 < ((2 * nhalo) + nj); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl18_q_im1[((__w0)*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)] = q[(((__w0 + ((nhalo - 1) - 0)))*(((nhalo + nj) + nhalo)) + (__w1))*(nk) + (__w2)];
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < (__inl18_hi - nhalo); ++__w0) {
          for (int64_t __w1 = 0; __w1 < __inl1_ny; ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl18_bl[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] = (__inl1_al[(((__w0 + (nhalo - 0)))*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)] - __inl18_q_i[((__w0)*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < ((__inl18_hi + 1) - (nhalo + 1)); ++__w0) {
          for (int64_t __w1 = 0; __w1 < __inl1_ny; ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl18_br[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] = (__inl1_al[(((__w0 + ((nhalo + 1) - 0)))*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)] - __inl18_q_i[((__w0)*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < (__inl18_hi - nhalo); ++__w0) {
          for (int64_t __w1 = 0; __w1 < __inl1_ny; ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl18_b0[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] = (__inl18_bl[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] + __inl18_br[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < ((__inl18_hi - 1) - (nhalo - 1)); ++__w0) {
          for (int64_t __w1 = 0; __w1 < __inl1_ny; ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl18_bl_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] = (__inl1_al[(((__w0 + ((nhalo - 1) - 0)))*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)] - __inl18_q_im1[((__w0)*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < (__inl18_hi - nhalo); ++__w0) {
          for (int64_t __w1 = 0; __w1 < __inl1_ny; ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl18_br_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] = (__inl1_al[(((__w0 + (nhalo - 0)))*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)] - __inl18_q_im1[((__w0)*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < ((__inl18_hi - 1) - (nhalo - 1)); ++__w0) {
          for (int64_t __w1 = 0; __w1 < __inl1_ny; ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl18_b0_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] = (__inl18_bl_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] + __inl18_br_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        if ((llabs(__inl1_ord_inner) == 5)) {
          for (int64_t __w0 = 0; __w0 < (__inl18_hi - nhalo); ++__w0) {
            for (int64_t __w1 = 0; __w1 < __inl1_ny; ++__w1) {
              for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
                __inl18_smt5[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] = ((__inl18_bl[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] * __inl18_br[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)]) < 0.0);
              }
            }
          }
          for (int64_t __w0 = 0; __w0 < ((__inl18_hi - 1) - (nhalo - 1)); ++__w0) {
            for (int64_t __w1 = 0; __w1 < __inl1_ny; ++__w1) {
              for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
                __inl18_smt5_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] = ((__inl18_bl_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] * __inl18_br_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)]) < 0.0);
              }
            }
          }
        }
        else {
          for (int64_t __w0 = 0; __w0 < (__inl18_hi - nhalo); ++__w0) {
            for (int64_t __w1 = 0; __w1 < __inl1_ny; ++__w1) {
              for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
                __inl18_smt5[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] = ((3.0 * fabs(__inl18_b0[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)])) < fabs((__inl18_bl[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] - __inl18_br[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)])));
              }
            }
          }
          for (int64_t __w0 = 0; __w0 < ((__inl18_hi - 1) - (nhalo - 1)); ++__w0) {
            for (int64_t __w1 = 0; __w1 < __inl1_ny; ++__w1) {
              for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
                __inl18_smt5_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] = ((3.0 * fabs(__inl18_b0_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)])) < fabs((__inl18_bl_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] - __inl18_br_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)])));
              }
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < (__inl18_hi - nhalo); ++__w0) {
          for (int64_t __w1 = 0; __w1 < __inl1_ny; ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl18_mask[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] = ((double)((__inl18_smt5[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)] | __inl18_smt5_m1[((__w0)*(__inl1_ny) + (__w1))*(nk) + (__w2)])));
            }
          }
        }
        /* numpy: np.where(__inl18_c > 0.0, __inl18_q_im1 + (1.0 - __inl18_c) * (__inl18_br_m1 - __inl18_c * __... */
        for (int64_t __r0 = 0; __r0 < (__inl18_hi - nhalo); ++__r0) {
          for (int64_t __r1 = 0; __r1 < ((2 * nhalo) + nj); ++__r1) {
            for (int64_t __r2 = 0; __r2 < nk; ++__r2) {
              __cb3[((__r0)*(((2 * nhalo) + nj)) + (__r1))*(nk) + (__r2)] = ((__inl18_c[((__r0)*(((2 * nhalo) + nj)) + (__r1))*(nk) + (__r2)] > 0.0) ? (__inl18_q_im1[((__r0)*(((2 * nhalo) + nj)) + (__r1))*(nk) + (__r2)] + (((1.0 - __inl18_c[((__r0)*(((2 * nhalo) + nj)) + (__r1))*(nk) + (__r2)]) * (__inl18_br_m1[((__r0)*(__inl1_ny) + (__r1))*(nk) + (__r2)] - (__inl18_c[((__r0)*(((2 * nhalo) + nj)) + (__r1))*(nk) + (__r2)] * __inl18_b0_m1[((__r0)*(__inl1_ny) + (__r1))*(nk) + (__r2)]))) * __inl18_mask[((__r0)*(__inl1_ny) + (__r1))*(nk) + (__r2)])) : (__inl18_q_i[((__r0)*(((2 * nhalo) + nj)) + (__r1))*(nk) + (__r2)] + (((1.0 + __inl18_c[((__r0)*(((2 * nhalo) + nj)) + (__r1))*(nk) + (__r2)]) * (__inl18_bl[((__r0)*(__inl1_ny) + (__r1))*(nk) + (__r2)] + (__inl18_c[((__r0)*(((2 * nhalo) + nj)) + (__r1))*(nk) + (__r2)] * __inl18_b0[((__r0)*(__inl1_ny) + (__r1))*(nk) + (__r2)]))) * __inl18_mask[((__r0)*(__inl1_ny) + (__r1))*(nk) + (__r2)])));
            }
          }
        }
        for (int64_t si0 = __inl18_lo; si0 < __inl18_hi; ++si0) {
          for (int64_t si1 = 0; si1 < ((2 * nhalo) + nj); ++si1) {
            for (int64_t si2 = 0; si2 < nk; ++si2) {
              __inl1_q_x_advected_mean[((si0)*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)] = __cb3[(((si0 - __inl18_lo))*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)];
            }
          }
        }
        __inl8_nx = ((nhalo + ni) + nhalo);
        __inl8_i0 = 3;
        __inl8_i1 = (__inl8_nx - 3);
        for (int64_t __w0 = 0; __w0 < (__inl8_i1 - __inl8_i0); ++__w0) {
          for (int64_t __w1 = 0; __w1 < ((2 * nhalo) + nj); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl8_fx1_i[((__w0)*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)] = (x_area_flux[(((__w0 + (__inl8_i0 - 0)))*(((nhalo + nj) + nhalo)) + (__w1))*(nk) + (__w2)] * __inl1_q_x_advected_mean[(((__w0 + (__inl8_i0 - 0)))*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < ((__inl8_i1 + 1) - (__inl8_i0 + 1)); ++__w0) {
          for (int64_t __w1 = 0; __w1 < ((2 * nhalo) + nj); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl8_fx1_ip1[((__w0)*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)] = (x_area_flux[(((__w0 + ((__inl8_i0 + 1) - 0)))*(((nhalo + nj) + nhalo)) + (__w1))*(nk) + (__w2)] * __inl1_q_x_advected_mean[(((__w0 + ((__inl8_i0 + 1) - 0)))*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < (__inl8_i1 - __inl8_i0); ++__w0) {
          for (int64_t __w1 = 0; __w1 < ((2 * nhalo) + nj); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl8_area_with_x_flux[((__w0)*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)] = ((area[(((__w0 + (__inl8_i0 - 0)))*(((nhalo + nj) + nhalo)) + (__w1))*(nk) + (__w2)] + x_area_flux[(((__w0 + (__inl8_i0 - 0)))*(((nhalo + nj) + nhalo)) + (__w1))*(nk) + (__w2)]) - x_area_flux[(((__w0 + ((__inl8_i0 + 1) - 0)))*(((nhalo + nj) + nhalo)) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t si0 = __inl8_i0; si0 < __inl8_i1; ++si0) {
          for (int64_t si1 = 0; si1 < ((2 * nhalo) + nj); ++si1) {
            for (int64_t si2 = 0; si2 < nk; ++si2) {
              __inl1_q_advected_x[((si0)*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)] = ((((q[(((si0 + (__inl8_i0 - __inl8_i0)))*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)] * area[(((si0 + (__inl8_i0 - __inl8_i0)))*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)]) + __inl8_fx1_i[(((si0 - __inl8_i0))*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)]) - __inl8_fx1_ip1[(((si0 - __inl8_i0))*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)]) / __inl8_area_with_x_flux[(((si0 - __inl8_i0))*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)]);
            }
          }
        }
        __inl19_j_end = ((nhalo + nj) - 1);
        __inl19_lo = (nhalo - 1);
        __inl19_hi = (__inl19_j_end + 3);
        for (int64_t si0 = 0; si0 < ((2 * nhalo) + ni); ++si0) {
          for (int64_t si1 = __inl19_lo; si1 < __inl19_hi; ++si1) {
            for (int64_t si2 = 0; si2 < nk; ++si2) {
              __inl1_al[((si0)*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)] = ((0.5833333333333334 * (__inl1_q_advected_x[((si0)*(((2 * nhalo) + nj)) + ((si1 + ((__inl19_lo - 1) - __inl19_lo))))*(nk) + (si2)] + __inl1_q_advected_x[((si0)*(((2 * nhalo) + nj)) + ((si1 + (__inl19_lo - __inl19_lo))))*(nk) + (si2)])) + (-0.08333333333333333 * (__inl1_q_advected_x[((si0)*(((2 * nhalo) + nj)) + ((si1 + ((__inl19_lo - 2) - __inl19_lo))))*(nk) + (si2)] + __inl1_q_advected_x[((si0)*(((2 * nhalo) + nj)) + ((si1 + ((__inl19_lo + 1) - __inl19_lo))))*(nk) + (si2)])));
            }
          }
        }
        if ((grid_type < 3)) {
          int64_t *__inl19_ja = (int64_t *)malloc(((2)) * sizeof(int64_t));
          __inl19_ja[0] = (nhalo - 1);
          __inl19_ja[1] = __inl19_j_end;
          for (int64_t __sc0 = 0; __sc0 < 2; ++__sc0) {
            for (int64_t __scs0 = 0; __scs0 < __inl1_nx; ++__scs0) {
              for (int64_t __scs1 = 0; __scs1 < nk; ++__scs1) {
                __inl1_al[((__scs0)*(((2 * nhalo) + nj)) + (__inl19_ja[__sc0]))*(nk) + (__scs1)] = (((-0.14285714285714285 * __inl1_q_advected_x[((__scs0)*(((2 * nhalo) + nj)) + ((__inl19_ja[__sc0] - 2)))*(nk) + (__scs1)]) + (0.7857142857142857 * __inl1_q_advected_x[((__scs0)*(((2 * nhalo) + nj)) + ((__inl19_ja[__sc0] - 1)))*(nk) + (__scs1)])) + (0.35714285714285715 * __inl1_q_advected_x[((__scs0)*(((2 * nhalo) + nj)) + (__inl19_ja[__sc0]))*(nk) + (__scs1)]));
              }
            }
          }
          int64_t *__inl19_jb = (int64_t *)malloc(((2)) * sizeof(int64_t));
          __inl19_jb[0] = nhalo;
          __inl19_jb[1] = (__inl19_j_end + 1);
          double *__inl19_left = (double *)malloc(((((2 * nhalo) + ni)) * (2) * (nk)) * sizeof(double));
          for (int64_t si0 = 0; si0 < ((2 * nhalo) + ni); ++si0) {
            for (int64_t si1 = 0; si1 < 2; ++si1) {
              for (int64_t si2 = 0; si2 < nk; ++si2) {
                __inl19_left[((si0)*(2) + (si1))*(nk) + (si2)] = (((((2.0 * dya[((si0)*(((nhalo + nj) + nhalo)) + ((__inl19_jb[si1] - 1)))*(nk) + (si2)]) + dya[((si0)*(((nhalo + nj) + nhalo)) + ((__inl19_jb[si1] - 2)))*(nk) + (si2)]) * __inl1_q_advected_x[((si0)*(((2 * nhalo) + nj)) + ((__inl19_jb[si1] - 1)))*(nk) + (si2)]) - (dya[((si0)*(((nhalo + nj) + nhalo)) + ((__inl19_jb[si1] - 1)))*(nk) + (si2)] * __inl1_q_advected_x[((si0)*(((2 * nhalo) + nj)) + ((__inl19_jb[si1] - 2)))*(nk) + (si2)])) / (dya[((si0)*(((nhalo + nj) + nhalo)) + ((__inl19_jb[si1] - 2)))*(nk) + (si2)] + dya[((si0)*(((nhalo + nj) + nhalo)) + ((__inl19_jb[si1] - 1)))*(nk) + (si2)]));
              }
            }
          }
          double *__inl19_right = (double *)malloc(((((2 * nhalo) + ni)) * (2) * (nk)) * sizeof(double));
          for (int64_t si0 = 0; si0 < ((2 * nhalo) + ni); ++si0) {
            for (int64_t si1 = 0; si1 < 2; ++si1) {
              for (int64_t si2 = 0; si2 < nk; ++si2) {
                __inl19_right[((si0)*(2) + (si1))*(nk) + (si2)] = (((((2.0 * dya[((si0)*(((nhalo + nj) + nhalo)) + (__inl19_jb[si1]))*(nk) + (si2)]) + dya[((si0)*(((nhalo + nj) + nhalo)) + ((__inl19_jb[si1] + 1)))*(nk) + (si2)]) * __inl1_q_advected_x[((si0)*(((2 * nhalo) + nj)) + (__inl19_jb[si1]))*(nk) + (si2)]) - (dya[((si0)*(((nhalo + nj) + nhalo)) + (__inl19_jb[si1]))*(nk) + (si2)] * __inl1_q_advected_x[((si0)*(((2 * nhalo) + nj)) + ((__inl19_jb[si1] + 1)))*(nk) + (si2)])) / (dya[((si0)*(((nhalo + nj) + nhalo)) + (__inl19_jb[si1]))*(nk) + (si2)] + dya[((si0)*(((nhalo + nj) + nhalo)) + ((__inl19_jb[si1] + 1)))*(nk) + (si2)]));
              }
            }
          }
          for (int64_t __scs0 = 0; __scs0 < __inl1_nx; ++__scs0) {
            for (int64_t __sc0 = 0; __sc0 < 2; ++__sc0) {
              for (int64_t __scs1 = 0; __scs1 < nk; ++__scs1) {
                __inl1_al[((__scs0)*(((2 * nhalo) + nj)) + (__inl19_jb[__sc0]))*(nk) + (__scs1)] = (0.5 * (__inl19_left[((__scs0)*(2) + (__sc0))*(nk) + (__scs1)] + __inl19_right[((__scs0)*(2) + (__sc0))*(nk) + (__scs1)]));
              }
            }
          }
          int64_t *__inl19_jc = (int64_t *)malloc(((2)) * sizeof(int64_t));
          __inl19_jc[0] = (nhalo + 1);
          __inl19_jc[1] = (__inl19_j_end + 2);
          for (int64_t __sc0 = 0; __sc0 < 2; ++__sc0) {
            for (int64_t __scs0 = 0; __scs0 < __inl1_nx; ++__scs0) {
              for (int64_t __scs1 = 0; __scs1 < nk; ++__scs1) {
                __inl1_al[((__scs0)*(((2 * nhalo) + nj)) + (__inl19_jc[__sc0]))*(nk) + (__scs1)] = (((0.35714285714285715 * __inl1_q_advected_x[((__scs0)*(((2 * nhalo) + nj)) + ((__inl19_jc[__sc0] - 1)))*(nk) + (__scs1)]) + (0.7857142857142857 * __inl1_q_advected_x[((__scs0)*(((2 * nhalo) + nj)) + (__inl19_jc[__sc0]))*(nk) + (__scs1)])) + (-0.14285714285714285 * __inl1_q_advected_x[((__scs0)*(((2 * nhalo) + nj)) + ((__inl19_jc[__sc0] + 1)))*(nk) + (__scs1)]));
              }
            }
          }
          free(__inl19_ja);
          free(__inl19_jb);
          free(__inl19_left);
          free(__inl19_right);
          free(__inl19_jc);
        }
        __inl20_j_end = ((nhalo + nj) - 1);
        __inl20_lo = nhalo;
        __inl20_hi = (__inl20_j_end + 2);
        for (int64_t __w0 = 0; __w0 < ((2 * nhalo) + ni); ++__w0) {
          for (int64_t __w1 = 0; __w1 < (__inl20_hi - nhalo); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl20_c[((__w0)*((__inl20_hi - nhalo)) + (__w1))*(nk) + (__w2)] = cry[((__w0)*(((nhalo + nj) + nhalo)) + ((__w1 + (nhalo - 0))))*(nk) + (__w2)];
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < ((2 * nhalo) + ni); ++__w0) {
          for (int64_t __w1 = 0; __w1 < (__inl20_hi - nhalo); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl20_q_j[((__w0)*((__inl20_hi - nhalo)) + (__w1))*(nk) + (__w2)] = __inl1_q_advected_x[((__w0)*(((2 * nhalo) + nj)) + ((__w1 + (nhalo - 0))))*(nk) + (__w2)];
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < ((2 * nhalo) + ni); ++__w0) {
          for (int64_t __w1 = 0; __w1 < ((__inl20_hi - 1) - (nhalo - 1)); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl20_q_jm1[((__w0)*(((__inl20_hi - 1) - (nhalo - 1))) + (__w1))*(nk) + (__w2)] = __inl1_q_advected_x[((__w0)*(((2 * nhalo) + nj)) + ((__w1 + ((nhalo - 1) - 0))))*(nk) + (__w2)];
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < __inl1_nx; ++__w0) {
          for (int64_t __w1 = 0; __w1 < (__inl20_hi - nhalo); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl20_bl[((__w0)*((__inl20_hi - nhalo)) + (__w1))*(nk) + (__w2)] = (__inl1_al[((__w0)*(((2 * nhalo) + nj)) + ((__w1 + (nhalo - 0))))*(nk) + (__w2)] - __inl20_q_j[((__w0)*((__inl20_hi - nhalo)) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < __inl1_nx; ++__w0) {
          for (int64_t __w1 = 0; __w1 < ((__inl20_hi + 1) - (nhalo + 1)); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl20_br[((__w0)*(((__inl20_hi + 1) - (nhalo + 1))) + (__w1))*(nk) + (__w2)] = (__inl1_al[((__w0)*(((2 * nhalo) + nj)) + ((__w1 + ((nhalo + 1) - 0))))*(nk) + (__w2)] - __inl20_q_j[((__w0)*((__inl20_hi - nhalo)) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < __inl1_nx; ++__w0) {
          for (int64_t __w1 = 0; __w1 < (__inl20_hi - nhalo); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl20_b0[((__w0)*((__inl20_hi - nhalo)) + (__w1))*(nk) + (__w2)] = (__inl20_bl[((__w0)*((__inl20_hi - nhalo)) + (__w1))*(nk) + (__w2)] + __inl20_br[((__w0)*(((__inl20_hi + 1) - (nhalo + 1))) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < __inl1_nx; ++__w0) {
          for (int64_t __w1 = 0; __w1 < ((__inl20_hi - 1) - (nhalo - 1)); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl20_bl_m1[((__w0)*(((__inl20_hi - 1) - (nhalo - 1))) + (__w1))*(nk) + (__w2)] = (__inl1_al[((__w0)*(((2 * nhalo) + nj)) + ((__w1 + ((nhalo - 1) - 0))))*(nk) + (__w2)] - __inl20_q_jm1[((__w0)*(((__inl20_hi - 1) - (nhalo - 1))) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < __inl1_nx; ++__w0) {
          for (int64_t __w1 = 0; __w1 < (__inl20_hi - nhalo); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl20_br_m1[((__w0)*((__inl20_hi - nhalo)) + (__w1))*(nk) + (__w2)] = (__inl1_al[((__w0)*(((2 * nhalo) + nj)) + ((__w1 + (nhalo - 0))))*(nk) + (__w2)] - __inl20_q_jm1[((__w0)*(((__inl20_hi - 1) - (nhalo - 1))) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < __inl1_nx; ++__w0) {
          for (int64_t __w1 = 0; __w1 < ((__inl20_hi - 1) - (nhalo - 1)); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl20_b0_m1[((__w0)*(((__inl20_hi - 1) - (nhalo - 1))) + (__w1))*(nk) + (__w2)] = (__inl20_bl_m1[((__w0)*(((__inl20_hi - 1) - (nhalo - 1))) + (__w1))*(nk) + (__w2)] + __inl20_br_m1[((__w0)*((__inl20_hi - nhalo)) + (__w1))*(nk) + (__w2)]);
            }
          }
        }
        if ((llabs(hord) == 5)) {
          for (int64_t __w0 = 0; __w0 < __inl1_nx; ++__w0) {
            for (int64_t __w1 = 0; __w1 < (__inl20_hi - nhalo); ++__w1) {
              for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
                __inl20_smt5[((__w0)*((__inl20_hi - nhalo)) + (__w1))*(nk) + (__w2)] = ((__inl20_bl[((__w0)*((__inl20_hi - nhalo)) + (__w1))*(nk) + (__w2)] * __inl20_br[((__w0)*(((__inl20_hi + 1) - (nhalo + 1))) + (__w1))*(nk) + (__w2)]) < 0.0);
              }
            }
          }
          for (int64_t __w0 = 0; __w0 < __inl1_nx; ++__w0) {
            for (int64_t __w1 = 0; __w1 < ((__inl20_hi - 1) - (nhalo - 1)); ++__w1) {
              for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
                __inl20_smt5_m1[((__w0)*(((__inl20_hi - 1) - (nhalo - 1))) + (__w1))*(nk) + (__w2)] = ((__inl20_bl_m1[((__w0)*(((__inl20_hi - 1) - (nhalo - 1))) + (__w1))*(nk) + (__w2)] * __inl20_br_m1[((__w0)*((__inl20_hi - nhalo)) + (__w1))*(nk) + (__w2)]) < 0.0);
              }
            }
          }
        }
        else {
          for (int64_t __w0 = 0; __w0 < __inl1_nx; ++__w0) {
            for (int64_t __w1 = 0; __w1 < (__inl20_hi - nhalo); ++__w1) {
              for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
                __inl20_smt5[((__w0)*((__inl20_hi - nhalo)) + (__w1))*(nk) + (__w2)] = ((3.0 * fabs(__inl20_b0[((__w0)*((__inl20_hi - nhalo)) + (__w1))*(nk) + (__w2)])) < fabs((__inl20_bl[((__w0)*((__inl20_hi - nhalo)) + (__w1))*(nk) + (__w2)] - __inl20_br[((__w0)*(((__inl20_hi + 1) - (nhalo + 1))) + (__w1))*(nk) + (__w2)])));
              }
            }
          }
          for (int64_t __w0 = 0; __w0 < __inl1_nx; ++__w0) {
            for (int64_t __w1 = 0; __w1 < ((__inl20_hi - 1) - (nhalo - 1)); ++__w1) {
              for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
                __inl20_smt5_m1[((__w0)*(((__inl20_hi - 1) - (nhalo - 1))) + (__w1))*(nk) + (__w2)] = ((3.0 * fabs(__inl20_b0_m1[((__w0)*(((__inl20_hi - 1) - (nhalo - 1))) + (__w1))*(nk) + (__w2)])) < fabs((__inl20_bl_m1[((__w0)*(((__inl20_hi - 1) - (nhalo - 1))) + (__w1))*(nk) + (__w2)] - __inl20_br_m1[((__w0)*((__inl20_hi - nhalo)) + (__w1))*(nk) + (__w2)])));
              }
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < __inl1_nx; ++__w0) {
          for (int64_t __w1 = 0; __w1 < (__inl20_hi - nhalo); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl20_mask[((__w0)*((__inl20_hi - nhalo)) + (__w1))*(nk) + (__w2)] = ((double)((__inl20_smt5[((__w0)*((__inl20_hi - nhalo)) + (__w1))*(nk) + (__w2)] | __inl20_smt5_m1[((__w0)*(((__inl20_hi - 1) - (nhalo - 1))) + (__w1))*(nk) + (__w2)])));
            }
          }
        }
        /* numpy: np.where(__inl20_c > 0.0, __inl20_q_jm1 + (1.0 - __inl20_c) * (__inl20_br_m1 - __inl20_c * __... */
        for (int64_t __r0 = 0; __r0 < ((2 * nhalo) + ni); ++__r0) {
          for (int64_t __r1 = 0; __r1 < (__inl20_hi - nhalo); ++__r1) {
            for (int64_t __r2 = 0; __r2 < nk; ++__r2) {
              __cb4[((__r0)*((nj + 1)) + (__r1))*(nk) + (__r2)] = ((__inl20_c[((__r0)*((__inl20_hi - nhalo)) + (__r1))*(nk) + (__r2)] > 0.0) ? (__inl20_q_jm1[((__r0)*(((__inl20_hi - 1) - (nhalo - 1))) + (__r1))*(nk) + (__r2)] + (((1.0 - __inl20_c[((__r0)*((__inl20_hi - nhalo)) + (__r1))*(nk) + (__r2)]) * (__inl20_br_m1[((__r0)*((__inl20_hi - nhalo)) + (__r1))*(nk) + (__r2)] - (__inl20_c[((__r0)*((__inl20_hi - nhalo)) + (__r1))*(nk) + (__r2)] * __inl20_b0_m1[((__r0)*(((__inl20_hi - 1) - (nhalo - 1))) + (__r1))*(nk) + (__r2)]))) * __inl20_mask[((__r0)*((__inl20_hi - nhalo)) + (__r1))*(nk) + (__r2)])) : (__inl20_q_j[((__r0)*((__inl20_hi - nhalo)) + (__r1))*(nk) + (__r2)] + (((1.0 + __inl20_c[((__r0)*((__inl20_hi - nhalo)) + (__r1))*(nk) + (__r2)]) * (__inl20_bl[((__r0)*((__inl20_hi - nhalo)) + (__r1))*(nk) + (__r2)] + (__inl20_c[((__r0)*((__inl20_hi - nhalo)) + (__r1))*(nk) + (__r2)] * __inl20_b0[((__r0)*((__inl20_hi - nhalo)) + (__r1))*(nk) + (__r2)]))) * __inl20_mask[((__r0)*((__inl20_hi - nhalo)) + (__r1))*(nk) + (__r2)])));
            }
          }
        }
        for (int64_t si0 = 0; si0 < ((2 * nhalo) + ni); ++si0) {
          for (int64_t si1 = __inl20_lo; si1 < __inl20_hi; ++si1) {
            for (int64_t si2 = 0; si2 < nk; ++si2) {
              __inl1_q_axya[((si0)*(((2 * nhalo) + nj)) + (si1))*(nk) + (si2)] = __cb4[((si0)*((nj + 1)) + ((si1 - __inl20_lo)))*(nk) + (si2)];
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < ((2 * nhalo) + ni); ++__w0) {
          for (int64_t __w1 = 0; __w1 < ((2 * nhalo) + nj); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl1_xuf[((__w0)*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)] = x_area_flux[((__w0)*(((nhalo + nj) + nhalo)) + (__w1))*(nk) + (__w2)];
            }
          }
        }
        for (int64_t __w0 = 0; __w0 < ((2 * nhalo) + ni); ++__w0) {
          for (int64_t __w1 = 0; __w1 < ((2 * nhalo) + nj); ++__w1) {
            for (int64_t __w2 = 0; __w2 < nk; ++__w2) {
              __inl1_yuf[((__w0)*(((2 * nhalo) + nj)) + (__w1))*(nk) + (__w2)] = y_area_flux[((__w0)*(((nhalo + nj) + nhalo)) + (__w1))*(nk) + (__w2)];
            }
          }
        }
        __inl10_i_end = ((nhalo + ni) - 1);
        __inl10_j_end = ((nhalo + nj) - 1);
        for (int64_t si0 = nhalo; si0 < (__inl10_i_end + 2); ++si0) {
          for (int64_t si1 = nhalo; si1 < (__inl10_j_end + 1); ++si1) {
            for (int64_t si2 = 0; si2 < nk; ++si2) {
              q_x_flux[((si0)*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)] = ((0.5 * (__inl1_q_ayxa[(((si0 + (nhalo - nhalo)))*(((2 * nhalo) + nj)) + ((si1 + (nhalo - nhalo))))*(nk) + (si2)] + __inl1_q_x_advected_mean[(((si0 + (nhalo - nhalo)))*(((2 * nhalo) + nj)) + ((si1 + (nhalo - nhalo))))*(nk) + (si2)])) * __inl1_xuf[(((si0 + (nhalo - nhalo)))*(((2 * nhalo) + nj)) + ((si1 + (nhalo - nhalo))))*(nk) + (si2)]);
            }
          }
        }
        for (int64_t si0 = nhalo; si0 < (__inl10_i_end + 1); ++si0) {
          for (int64_t si1 = nhalo; si1 < (__inl10_j_end + 2); ++si1) {
            for (int64_t si2 = 0; si2 < nk; ++si2) {
              q_y_flux[((si0)*(((nhalo + nj) + nhalo)) + (si1))*(nk) + (si2)] = ((0.5 * (__inl1_q_axya[(((si0 + (nhalo - nhalo)))*(((2 * nhalo) + nj)) + ((si1 + (nhalo - nhalo))))*(nk) + (si2)] + __inl1_q_y_advected_mean[(((si0 + (nhalo - nhalo)))*(((2 * nhalo) + nj)) + ((si1 + (nhalo - nhalo))))*(nk) + (si2)])) * __inl1_yuf[(((si0 + (nhalo - nhalo)))*(((2 * nhalo) + nj)) + ((si1 + (nhalo - nhalo))))*(nk) + (si2)]);
            }
          }
        }
        free(__inl1_q_y_advected_mean);
        free(__inl1_q_x_advected_mean);
        free(__inl1_q_advected_y);
        free(__inl1_q_advected_x);
        free(__inl1_q_ayxa);
        free(__inl1_q_axya);
        free(__inl1_al);
        free(__cb1);
        free(__cb2);
        free(__cb3);
        free(__cb4);
        free(__inl14_bl);
        free(__inl14_br);
        free(__inl14_b0);
        free(__inl14_bl_m1);
        free(__inl14_br_m1);
        free(__inl14_b0_m1);
        free(__inl14_smt5);
        free(__inl14_smt5_m1);
        free(__inl4_fyy_j);
        free(__inl4_fyy_jp1);
        free(__inl4_denom);
        free(__inl16_bl);
        free(__inl16_br);
        free(__inl16_b0);
        free(__inl16_bl_m1);
        free(__inl16_br_m1);
        free(__inl16_b0_m1);
        free(__inl16_smt5);
        free(__inl16_smt5_m1);
        free(__inl18_bl);
        free(__inl18_br);
        free(__inl18_b0);
        free(__inl18_bl_m1);
        free(__inl18_br_m1);
        free(__inl18_b0_m1);
        free(__inl18_smt5);
        free(__inl18_smt5_m1);
        free(__inl8_fx1_i);
        free(__inl8_fx1_ip1);
        free(__inl8_area_with_x_flux);
        free(__inl20_bl);
        free(__inl20_br);
        free(__inl20_b0);
        free(__inl20_bl_m1);
        free(__inl20_br_m1);
        free(__inl20_b0_m1);
        free(__inl20_smt5);
        free(__inl20_smt5_m1);
        free(__inl1_xuf);
        free(__inl1_yuf);
        free(__inl14_c);
        free(__inl14_q_j);
        free(__inl14_q_jm1);
        free(__inl16_c);
        free(__inl16_q_i);
        free(__inl16_q_im1);
        free(__inl18_c);
        free(__inl18_q_i);
        free(__inl18_q_im1);
        free(__inl20_c);
        free(__inl20_q_j);
        free(__inl20_q_jm1);
        free(__inl14_mask);
        free(__inl16_mask);
        free(__inl18_mask);
        free(__inl20_mask);
}
