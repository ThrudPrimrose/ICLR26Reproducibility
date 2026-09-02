// hpcagent_bench-autogen -- generated from spgemm_hash_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
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

static double _select_bin(const double size) {
    int64_t chosen;
    int64_t low;
    int64_t high;
    chosen = -1;
    low = 0;
    high = 32;
    for (int64_t b = 0; b < 8; ++b) {
      if (((size > low) && (size <= high))) {
        chosen = b;
      }
      low = high;
      high = (high * 2);
    }
    return chosen;
}

static double _table_size(const double b) {
    int64_t ts;
    ts = 32;
    for (int64_t _ = 0; _ < b; ++_) {
      ts = (ts * 2);
    }
    return ts;
}

static void _bitonic_sort(int64_t *restrict key, const double n) {
    int64_t size;
    int64_t stride;
    int64_t pos;
    int64_t left;
    int64_t right;
    int64_t ascending;
    int64_t greater;
    size = 2;
    while ((size < n)) {
      stride = int_floor(size, 2);
      while ((stride > 0)) {
        for (int64_t idx = 0; idx < int_floor(n, 2); ++idx) {
          ascending = 1;
          if ((python_mod(int_floor(idx, int_floor(size, 2)), 2) == 1)) {
            ascending = 0;
          }
          pos = ((2 * idx) - python_mod(idx, stride));
          left = key[pos];
          right = key[(pos + stride)];
          greater = 0;
          if ((left > right)) {
            greater = 1;
          }
          if ((greater == ascending)) {
            key[pos] = right;
            key[(pos + stride)] = left;
          }
        }
        stride = int_floor(stride, 2);
      }
      size = (size * 2);
    }
    stride = int_floor(n, 2);
    while ((stride > 0)) {
      for (int64_t idx = 0; idx < int_floor(n, 2); ++idx) {
        pos = ((2 * idx) - python_mod(idx, stride));
        left = key[pos];
        right = key[(pos + stride)];
        if ((left > right)) {
          key[pos] = right;
          key[(pos + stride)] = left;
        }
      }
      stride = int_floor(stride, 2);
    }
}

static double _select_bin__s2(const double size) {
    int64_t chosen;
    int64_t low;
    int64_t high;
    chosen = -1;
    low = 0;
    high = 32;
    for (int64_t b = 0; b < 8; ++b) {
      if (((size > low) && (size <= high))) {
        chosen = b;
      }
      low = high;
      high = (high * 2);
    }
    return chosen;
}

void spgemm_hash_fp64(const int64_t *restrict A_indices, const int64_t *restrict A_indptr, const int64_t *restrict B_indices, const int64_t *restrict B_indptr, int64_t *restrict C_indices, int64_t *restrict C_indptr, const int64_t K, const int64_t M, const int64_t N, const int64_t nnz_A, const int64_t nnz_B, const int64_t nnz_C_cap) {
        int64_t running;
        int64_t products;
        int64_t chosen;
        int64_t row;
        int64_t a_col;
        int64_t ts;
        int64_t distinct;
        int64_t base;
        int64_t count;
        int64_t b_col;
        int64_t slot;
        int64_t probing;
        int64_t held;
        int64_t *prod = calloc((size_t)M, sizeof(int64_t));
        /* removed memset: calloc already zeroes */
        int64_t *row_bin = calloc((size_t)M, sizeof(int64_t));
        /* removed memset: calloc already zeroes */
        int64_t *row_nnz = calloc((size_t)M, sizeof(int64_t));
        /* removed memset: calloc already zeroes */
        int64_t bin_size[8] = {0};
        /* removed memset: initialized statically */
        int64_t bin_offset[8] = {0};
        /* removed memset: initialized statically */
        int64_t *rows_in_bins = malloc((size_t)M * sizeof(int64_t));
        /* rows_in_bins will be initialized later */
        int64_t table[(4096)];
        /* removed memset: calloc already zeroes */
        /* removed memset: calloc already zeroes */
        /* removed memset: calloc already zeroes */
        /* removed memset: initialized statically */
        /* removed memset: initialized statically */
        /* rows_in_bins will be initialized later */
        for (int64_t i = 0; i < M; ++i) {
          products = 0;
          for (int64_t j = A_indptr[i]; j < A_indptr[(i + 1)]; ++j) {
            a_col = A_indices[j];
            products = (products + (B_indptr[(a_col + 1)] - B_indptr[a_col]));
          }
          if ((products > N)) {
            products = N;
          }
          prod[i] = products;
        }
        for (int64_t b = 0; b < 8; ++b) {
          bin_size[b] = 0;
        }
        for (int64_t i = 0; i < M; ++i) {
          chosen = _select_bin(prod[i]);
          row_bin[i] = chosen;
          if ((chosen >= 0)) {
            bin_size[chosen] = (bin_size[chosen] + 1);
          }
        }
        running = 0;
        for (int64_t b = 0; b < 8; ++b) {
          bin_offset[b] = running;
          running = (running + bin_size[b]);
          bin_size[b] = 0;
        }
        #pragma omp parallel for schedule(dynamic) private(table)
        for (int64_t r = 0; r < M; ++r) {
          rows_in_bins[r] = -1;
        }
        for (int64_t i = 0; i < M; ++i) {
          chosen = row_bin[i];
          if ((chosen >= 0)) {
            rows_in_bins[(bin_offset[chosen] + bin_size[chosen])] = i;
            bin_size[chosen] = (bin_size[chosen] + 1);
          }
        }
        #pragma omp parallel for schedule(dynamic) private(table)
        for (int64_t r = 0; r < M; ++r) {
          row = rows_in_bins[r];
          if ((row >= 0)) {
            ts = _table_size(row_bin[row]);
            for (int64_t t = 0; t < ts; ++t) {
              table[t] = N;
            }
            distinct = 0;
            for (int64_t j = A_indptr[row]; j < A_indptr[(row + 1)]; ++j) {
              a_col = A_indices[j];
              for (int64_t k = B_indptr[a_col]; k < B_indptr[(a_col + 1)]; ++k) {
                b_col = B_indices[k];
                slot = python_mod((b_col * 107), ts);
                probing = 1;
                while ((probing == 1)) {
                  held = table[slot];
                  if ((held == b_col)) {
                    probing = 0;
                  }
                  else if ((held == N)) {
                    table[slot] = b_col;
                    distinct = (distinct + 1);
                    probing = 0;
                  }
                  else {
                    slot = (slot + 1);
                    if ((slot == ts)) {
                      slot = 0;
                    }
                  }
                }
              }
            }
            row_nnz[row] = distinct;
          }
        }
        running = 0;
        for (int64_t i = 0; i < M; ++i) {
          C_indptr[i] = running;
          running = (running + row_nnz[i]);
        }
        C_indptr[M] = running;
        for (int64_t b = 0; b < 8; ++b) {
          bin_size[b] = 0;
        }
        for (int64_t i = 0; i < M; ++i) {
          chosen = _select_bin__s2(row_nnz[i]);
          row_bin[i] = chosen;
          if ((chosen >= 0)) {
            bin_size[chosen] = (bin_size[chosen] + 1);
          }
        }
        running = 0;
        for (int64_t b = 0; b < 8; ++b) {
          bin_offset[b] = running;
          running = (running + bin_size[b]);
          bin_size[b] = 0;
        }
        #pragma omp parallel for schedule(dynamic) private(table)
        for (int64_t r = 0; r < M; ++r) {
          rows_in_bins[r] = -1;
        }
        for (int64_t i = 0; i < M; ++i) {
          chosen = row_bin[i];
          if ((chosen >= 0)) {
            rows_in_bins[(bin_offset[chosen] + bin_size[chosen])] = i;
            bin_size[chosen] = (bin_size[chosen] + 1);
          }
        }
        #pragma omp parallel for schedule(dynamic) private(table)
        for (int64_t r = 0; r < M; ++r) {
          row = rows_in_bins[r];
          if ((row >= 0)) {
            ts = _table_size(row_bin[row]);
            for (int64_t t = 0; t < ts; ++t) {
              table[t] = N;
            }
            for (int64_t j = A_indptr[row]; j < A_indptr[(row + 1)]; ++j) {
              a_col = A_indices[j];
              for (int64_t k = B_indptr[a_col]; k < B_indptr[(a_col + 1)]; ++k) {
                b_col = B_indices[k];
                slot = python_mod((b_col * 107), ts);
                probing = 1;
                while ((probing == 1)) {
                  held = table[slot];
                  if ((held == b_col)) {
                    probing = 0;
                  }
                  else if ((held == N)) {
                    table[slot] = b_col;
                    probing = 0;
                  }
                  else {
                    slot = (slot + 1);
                    if ((slot == ts)) {
                      slot = 0;
                    }
                  }
                }
              }
            }
            _bitonic_sort(table, ts);
            base = C_indptr[row];
            count = (C_indptr[(row + 1)] - base);
            for (int64_t t = 0; t < count; ++t) {
              C_indices[(base + t)] = table[t];
            }
          }
        }
        free(prod);
        free(row_bin);
        free(row_nnz);
        free(rows_in_bins);
}
