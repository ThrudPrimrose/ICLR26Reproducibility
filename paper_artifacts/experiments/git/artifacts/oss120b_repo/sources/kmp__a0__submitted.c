// SPDX-License-Identifier: MIT
#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <alloca.h>
#include <omp.h>

#ifndef MAX_PATTERN_SIZE
#define MAX_PATTERN_SIZE 4096
#endif
static int64_t fail_buf[MAX_PATTERN_SIZE];
static int64_t next0_buf[MAX_PATTERN_SIZE];
static int64_t next1_buf[MAX_PATTERN_SIZE];
static inline double _Complex __npb_conj(double _Complex z) { return conj(z); }
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif
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
#define __NPB_UNSIGNED_ASSOC(fn) \
    unsigned int: fn, unsigned long: fn, unsigned long long: fn,
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

void kmp_fp64(int64_t *restrict matches, const int64_t *restrict pattern, const int64_t *restrict text, const int64_t M, const int64_t N) {
    // Build failure function (prefix function) for KMP
    int64_t *restrict fail = fail_buf;
    fail[0] = 0;
    int64_t k = 0;
    for (int64_t i = 1; i < M; ++i) {
        while (k > 0 && pattern[k] != pattern[i]) {
            k = fail[k - 1];
        }
        if (pattern[k] == pattern[i]) {
            ++k;
        }
        fail[i] = k;
    }
    // Build deterministic transition tables for binary alphabet {0,1}
    // next0[q] = next state when current symbol is 0 and we are in state q
    // next1[q] = next state when current symbol is 1 and we are in state q
    int64_t *restrict next0 = next0_buf;
    int64_t *restrict next1 = next1_buf;
    for (int64_t q = 0; q < M; ++q) {
        // transition on symbol 0
        if (pattern[q] == 0) {
            next0[q] = q + 1;
        } else {
            next0[q] = (q == 0) ? 0 : next0[fail[q - 1]];
        }
        // transition on symbol 1
        if (pattern[q] == 1) {
            next1[q] = q + 1;
        } else {
            next1[q] = (q == 0) ? 0 : next1[fail[q - 1]];
        }
    }
    // Scan the text using the pre‑computed tables – use sequential loop for small N, else parallelized
    int64_t count = 0;
    if (N < 1000000) {
        int64_t q_state = 0;
        for (int64_t i = 0; i < N; ++i) {
            int64_t sym = text[i];
            if (sym == 0) {
                q_state = next0[q_state];
            } else if (sym == 1) {
                q_state = next1[q_state];
            } else {
                // Fallback for non‑binary symbols – use classic KMP step
                while (q_state > 0 && pattern[q_state] != sym) {
                    q_state = fail[q_state - 1];
                }
                if (pattern[q_state] == sym) {
                    ++q_state;
                }
            }
            if (q_state == M) {
                ++count;
                q_state = fail[M - 1];
            }
        }
    } else {
        #pragma omp parallel
        {
            int64_t local_count = 0;
            int64_t q_state = 0;
            int64_t thread_id = omp_get_thread_num();
            int64_t num_threads = omp_get_num_threads();
            int64_t chunk = (N + num_threads - 1) / num_threads;
            int64_t start = thread_id * chunk;
            int64_t end = start + chunk;
            if (end > N) end = N;
            int64_t overlap_start = start > (M - 1) ? start - (M - 1) : 0;
            for (int64_t i = overlap_start; i < end; ++i) {
                int64_t sym = text[i];
                if (sym == 0) {
                    q_state = next0[q_state];
                } else if (sym == 1) {
                    q_state = next1[q_state];
                } else {
                    // Fallback for non‑binary symbols – use classic KMP step
                    while (q_state > 0 && pattern[q_state] != sym) {
                        q_state = fail[q_state - 1];
                    }
                    if (pattern[q_state] == sym) {
                        ++q_state;
                    }
                }
                if (q_state == M) {
                    int64_t match_start = i - M + 1;
                    if (match_start >= start) {
                        ++local_count;
                    }
                    q_state = fail[M - 1];
                }
            }
            #pragma omp atomic
            count += local_count;
        }
    }
    matches[0] = count;
    return;
}
