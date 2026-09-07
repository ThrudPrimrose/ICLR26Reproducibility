// hpcagent_bench-autogen -- generated from kmp_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <immintrin.h>
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
 * second operand). ``(a)+(b)`` is NaN whenever either is; for finite
 * operands the ternary picks the larger/smaller -- identical to a plain
 * comparison, so the 3-way builtin max (needleman_wunsch, always finite)
 * is unchanged. For integer operands the NaN test is dead. */
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
    float: __npb_ceildiv_f, double: __npb_floordiv_f, long double: __npb_ceildiv_f, \
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

static void build_fail(const int64_t *restrict pattern, int64_t M, int64_t *restrict fail) {
    fail[0] = 0;
    int64_t k = 0;
    for (int64_t i = 1; i < M; ++i) {
        while (k > 0 && pattern[k] != pattern[i]) {
            k = fail[k - 1];
        }
        if (pattern[k] == pattern[i]) {
            k += 1;
        }
        fail[i] = k;
    }
}

/* Process text[lo:hi] with KMP starting from state q0, count matches whose end
 * position i satisfies i >= start_count.  Returns final state q. */
static inline int64_t kmp_process(const int64_t *restrict pat,
                                  const int64_t *restrict txt,
                                  const int64_t *restrict fl,
                                  int64_t M, int64_t lo, int64_t hi,
                                  int64_t start_count, int64_t q0,
                                  int64_t *restrict count_out) {
    int64_t q = q0;
    int64_t count = 0;
    int64_t i = lo;
    const __m512i vpat0 = _mm512_set1_epi64(pat[0]);

    while (i < hi) {
        if (__builtin_expect(q == 0 && i + 8 <= hi, 1)) {
            __m512i vtxt = _mm512_loadu_si512((__m512i const *)(txt + i));
            __mmask8 mask = _mm512_cmpeq_epi64_mask(vtxt, vpat0);
            if (mask == 0) {
                i += 8;
                continue;
            }
            int k = __builtin_ctz((unsigned)mask);
            i += k;
            if (M == 1) {
                if (i >= start_count) count++;
                i++;
                continue;
            }
            q = 1;
            i++;
            continue;
        }

        const int64_t c = txt[i];
        while (q > 0 && pat[q] != c) {
            q = fl[q - 1];
        }
        if (pat[q] == c) {
            q += 1;
        }
        if (q == M) {
            if (i >= start_count) count++;
            q = fl[q - 1];
        }
        i++;
    }

    *count_out = count;
    return q;
}

void kmp_fp64(int64_t *restrict matches, const int64_t *restrict pattern, const int64_t *restrict text, const int64_t M, const int64_t N, uint8_t *restrict workspace, const int64_t workspace_size) {
    (void)workspace_size;
    int64_t *fail;
    if (workspace && (size_t)workspace_size >= (size_t)M * sizeof(int64_t)) {
        fail = (int64_t *)workspace;
    } else {
        fail = (int64_t *)malloc((size_t)M * sizeof(int64_t));
    }
    memset(fail, 0, (size_t)M * sizeof(int64_t));

    build_fail(pattern, M, fail);

    int64_t count = 0;

    const int64_t *pat = pattern;
    const int64_t *txt = text;
    const int64_t *fl = fail;

    int max_threads = omp_get_max_threads();
    int use_threads = 1;
    if (max_threads > 1 && N > 10000 && M > 0 && M < N / 2) {
        long long t = (long long)N / (5LL * (long long)M);
        if (t < 1) t = 1;
        if (t > max_threads) t = max_threads;
        use_threads = (int)t;
    }

    if (use_threads <= 1) {
        int64_t unused_q;
        count = 0;
        unused_q = kmp_process(pat, txt, fl, M, 0, N, 0, 0, &count);
        (void)unused_q;
    } else {
        #pragma omp parallel num_threads(use_threads) reduction(+:count)
        {
            int tid = omp_get_thread_num();
            int64_t chunk = (N + use_threads - 1) / use_threads;
            int64_t start = (int64_t)tid * chunk;
            int64_t end = start + chunk;
            if (end > N) end = N;
            if (start < end) {
                int64_t lo = (start > M - 1) ? start - (M - 1) : 0;
                int64_t q;
                int64_t local_count = 0;
                q = kmp_process(pat, txt, fl, M, lo, start, start, 0, &local_count);
                q = kmp_process(pat, txt, fl, M, start, end, start, q, &local_count);
                (void)q;
                count += local_count;
            }
        }
    }

    matches[0] = count;
    if (fail != (int64_t *)workspace) {
        free(fail);
    }
}
