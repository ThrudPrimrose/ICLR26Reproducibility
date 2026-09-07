/* KMP substring search: count (overlapping) occurrences of `pattern` in `text`.
 *
 * The reference algorithm is classic KMP with a prefix-failure table.  This
 * implementation keeps exactly the same observable result (matches[0] = number
 * of overlapping occurrences) while replacing the serial failure-function scan
 * with a bit-parallel Shift-And automaton:
 *
 *   state R (64 bit): bit j set  <=>  pattern[0..j] equals the suffix of the
 *                                     text processed so far
 *   R <- ((R << 1) | 1) & R_{text[i]},   match when bit (M-1) of R is set
 *
 * R_{0}/R_{1} are two 64-bit masks built from the (binary) pattern, so one
 * text word costs a few register ops with no memory dependency and no loop
 * branch.  Eight independent automata are packed into the 64-bit lanes of one
 * AVX-512 register (four in AVX-2, one scalar otherwise).
 *
 * Parallelism: thread t owns start positions [s,e).  The automaton state at
 * position i only depends on text[i-M+1..i], so starting the automaton from
 * 0 at max(0, s-(M-1)) makes every state at i >= s+(M-1) identical to the
 * global one.  The thread therefore processes [max(0,s-M+1), min(N,e+M-1))
 * and counts only end positions in [s+M-1, min(N,e+M-1)).  Adjacent threads
 * thus count a disjoint, gap-free partition of all occurrences.  Extra work
 * is 2*(M-1) words per thread -- negligible for the short patterns this
 * kernel is used with.  For M > 64 (fits no 64-bit state) the identical
 * context trick is applied to plain KMP, which remains exactly correct for
 * arbitrary 64-bit values.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#define KMP_HAS_512 1
#else
#define KMP_HAS_512 0
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#define KMP_HAS_256 1
#else
#define KMP_HAS_256 0
#endif

/* ------------------------------------------------------------------ */
/* Serial building blocks (scalar).                                    */

/* One Shift-And step.  c must be 0 or 1 on the fast path; the 3-way form
 * below is the exact one used when the pattern (hence the text) is not a
 * binary word sequence. */
static inline uint64_t sa_step(uint64_t R, uint64_t R0, uint64_t R1, int64_t c) {
    uint64_t Rsel = (c == 0) ? R0 : (c == 1) ? R1 : 0;
    return ((R << 1) | 1) & Rsel;
}

/* Shift-And over text[a,b); count matches ending at i in [ca,cb).
 * Exact for arbitrary int64 values. */
static int64_t sa_range(const int64_t *restrict text,
                        uint64_t R0, uint64_t R1, int64_t matchbit,
                        int64_t a, int64_t b, int64_t ca, int64_t cb) {
    uint64_t R = 0;
    int64_t cnt = 0;
    for (int64_t i = a; i < b; ++i) {
        R = sa_step(R, R0, R1, text[i]);
        if (i >= ca && i < cb)
            cnt += (int64_t)((R >> matchbit) & 1);
    }
    return cnt;
}

/* KMP over text[a,b) with failure table `fail`; count matches ending at i
 * in [ca,cb).  Exact for arbitrary values, any M. */
static int64_t kmp_range(const int64_t *restrict pattern,
                         const int64_t *restrict fail,
                         const int64_t *restrict text, int64_t M,
                         int64_t a, int64_t b, int64_t ca, int64_t cb) {
    int64_t q = 0, cnt = 0;
    for (int64_t i = a; i < b; ++i) {
        int64_t c = text[i];
        while (q > 0 && pattern[q] != c)
            q = fail[q - 1];
        if (pattern[q] == c)
            q += 1;
        if (q == M) {
            if (i >= ca && i < cb)
                cnt += 1;
            q = fail[q - 1];
        }
    }
    return cnt;
}

/* ------------------------------------------------------------------ */
/* SIMD Shift-And bodies.  All lanes must see binary (0/1) text.       */

#if KMP_HAS_512
static int64_t sa512_range(const int64_t *restrict text,
                           uint64_t R0, uint64_t R1, uint64_t BM,
                           int64_t a, int64_t b) {
    __m512i R    = _mm512_setzero_si512();
    const __m512i R0v = _mm512_set1_epi64((int64_t)R0);
    const __m512i DX  = _mm512_set1_epi64((int64_t)(R1 ^ R0));
    const __m512i ONE = _mm512_set1_epi64(1);
    const __m512i BMv = _mm512_set1_epi64((int64_t)BM);
    const __m512i ZER = _mm512_setzero_si512();
    int64_t cnt = 0;
    int64_t i = a;
    for (; i + 8 <= b; i += 8) {
        _mm_prefetch((const char *)(text + i + 128), _MM_HINT_T0);
        __m512i c   = _mm512_loadu_si512(text + i);
        __m512i sel = _mm512_or_si512(R0v, _mm512_and_si512(DX, c));
        R = _mm512_and_si512(_mm512_or_si512(_mm512_slli_epi64(R, 1), ONE), sel);
        cnt += (int64_t)_cpop_u8(_mm512_cmpneq_epi64_mask(R & BMv, ZER));
    }
    return cnt;
}
#endif

#if KMP_HAS_256
static int64_t sa256_range(const int64_t *restrict text,
                           uint64_t R0, uint64_t R1, uint64_t BM,
                           int64_t a, int64_t b) {
    __m256i R    = _mm256_setzero_si256();
    const __m256i R0v = _mm256_set1_epi64((int64_t)R0);
    const __m256i DX  = _mm256_set1_epi64((int64_t)(R1 ^ R0));
    const __m256i ONE = _mm256_set1_epi64(1);
    const __m256i BMv = _mm256_set1_epi64((int64_t)BM);
    int64_t cnt = 0;
    int64_t i = a;
    for (; i + 4 <= b; i += 4) {
        _mm_prefetch((const char *)(text + i + 64), _MM_HINT_T0);
        __m256i c   = _mm256_loadu_si256(text + i);
        __m256i sel = _mm256_or_si256(R0v, _mm256_and_si256(DX, c));
        R = _mm256_and_si256(_mm256_or_si256(_mm256_slli_epi64(R, 1), ONE), sel);
        cnt += (int64_t)__builtin_popcount8(_mm256_movemask_epi8(R & BMv) & 0x1111);
    }
    return cnt;
}
#endif

/* Shift-And over text[a,b) with 8-lane SIMD body (binary text).
 * Returns (count, state after b).  a may be unaligned; b is whatever the
 * caller chose. */
static int64_t sa_simd_range(const int64_t *restrict text,
                             uint64_t R0, uint64_t R1, uint64_t BM,
                             int64_t a, int64_t b, uint64_t *Rout) {
    uint64_t R = 0;
    int64_t cnt = 0;
    int64_t i = a;
    for (; i < b && (i & 7); ++i)
        R = sa_step(R, R0, R1, text[i]);
#if KMP_HAS_512
    {
        int64_t j = i + 8 * ((b - i) >> 3);
        if (j > i) {
            __m512i Rv    = _mm512_setzero_si512();
            const __m512i R0v = _mm512_set1_epi64((int64_t)R0);
            const __m512i DX  = _mm512_set1_epi64((int64_t)(R1 ^ R0));
            const __m512i ONE = _mm512_set1_epi64(1);
            const __m512i BMv = _mm512_set1_epi64((int64_t)BM);
            const __m512i ZER = _mm512_setzero_si512();
            for (; i < j; i += 8) {
                _mm_prefetch((const char *)(text + i + 128), _MM_HINT_T0);
                __m512i c   = _mm512_loadu_si512(text + i);
                __m512i sel = _mm512_or_si512(R0v, _mm512_and_si512(DX, c));
                Rv = _mm512_and_si512(_mm512_or_si512(_mm512_slli_epi64(Rv, 1), ONE), sel);
                cnt += (int64_t)_cpop_u8(_mm512_cmpneq_epi64_mask(Rv & BMv, ZER));
            }
            R = (uint64_t)_mm512_extract_epi64(Rv, 0);
        }
    }
#elif KMP_HAS_256
    {
        int64_t j = i + 4 * ((b - i) >> 2);
        for (; i < j; i += 4) {
            _mm_prefetch((const char *)(text + i + 64), _MM_HINT_T0);
            __m256i c   = _mm256_loadu_si256(text + i);
            __m256i sel = _mm256_or_si256(R0v... 
        }
    }
#endif
    for (; i < b; ++i)
        R = sa_step(R, R0, R1, text[i]);
    *Rout = R;
    return cnt;
}
