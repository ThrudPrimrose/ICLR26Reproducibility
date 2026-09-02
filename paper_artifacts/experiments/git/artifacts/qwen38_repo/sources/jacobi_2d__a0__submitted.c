// Optimized 2-D Jacobi stencil (5-point, two sweeps per timestep).
//
// Numerics: bit-identical to the NumPy reference. Per element
//   0.2 * ((((c + l) + r) + d) + u)
// in the same left-associative order. Only the element ordering is
// parallelized (OpenMP rows / AVX-512 8-wide lanes).
//
// Fast path: a 3-output-row window keeps five consecutive input rows in
// register fragments (prev / current / next 8-wide), so each row is
// fetched ~1.67x instead of 3x and three output rows are produced per
// pass. Falls back to the auto-vectorized row path without AVX-512.

#include <stdint.h>
#include <immintrin.h>

static inline void row_simple(const double *restrict src, double *restrict dst,
                              const int64_t N) {
    const double *restrict up = src;
    const double *restrict cn = src + N;
    const double *restrict dn = src + 2 * N;
    double *restrict o = dst + N;
    for (int64_t j = 1; j < N - 1; ++j)
        o[j] = 0.2 * ((((cn[j] + cn[j - 1]) + cn[j + 1]) + dn[j]) + up[j]);
}

static void sweep_simple(const double *restrict src, double *restrict dst,
                         const int64_t N, const int64_t i0, const int64_t i1) {
    for (int64_t i = i0; i < i1; ++i)
        row_simple(src + (i - 1) * N, dst + (i - 1) * N, N);
}

// ------------------------------------------------------------- AVX-512 core
static __attribute__((target("avx512f"))) inline __m512d
out5(const __m512d up, const __m512d cL, const __m512d c, const __m512d cR,
     const __m512d dn) {
    __m512d t = _mm512_add_pd(c, cL);
    t = _mm512_add_pd(t, cR);
    t = _mm512_add_pd(t, dn);
    t = _mm512_add_pd(t, up);
    return _mm512_mul_pd(_mm512_set1_pd(0.2), t);
}

// Wrap-around neighbor vectors over the concat (hi lanes 0..7, lo lanes 8..15):
//   L = [hi7, lo0..lo6]      R = [lo1..lo7, hi0]
// VPERMVPD wraps modulo 16 (VPERMPD would zero-fill the top lanes).
static __attribute__((target("avx512f"))) inline __m512i
mkidxL(void) {
    return _mm512_setr_epi8(7, 8, 9, 10, 11, 12, 13, 14,
                            0, 0, 0, 0, 0, 0, 0, 0);
}
static __attribute__((target("avx512f"))) inline __m512i
mkidxR(void) {
    return _mm512_setr_epi8(9, 10, 11, 12, 13, 14, 15, 0,
                            0, 0, 0, 0, 0, 0, 0, 0);
}
static __attribute__((target("avx512f"))) inline __m512d
shiftL(const __m512d hi, const __m512d lo, const __m512i idx) {
    const __m512d cat = _mm512_mask_blend_pd(0xFF, hi, lo);
    return _mm512_permvar_pd(cat, idx);
}
static __attribute__((target("avx512f"))) inline __m512d
shiftR(const __m512d hi, const __m512d lo, const __m512i idx) {
    const __m512d cat = _mm512_mask_blend_pd(0xFF, hi, lo);
    return _mm512_permvar_pd(cat, idx);
}

// Output rows r..r+2 from input rows r-1..r+3 (all interior-safe:
// r >= 1, r+2 <= N-2 => rows 0..N-1 exist). Tail window (1-2 rows)
// takes the scalar path.
static __attribute__((target("avx512f"))) void
window3(const double *restrict src, double *restrict dst, const int64_t N,
        const int64_t r) {
    const int64_t J = N - 2;
    if (r + 2 > N - 2) {
        for (int64_t i = r; i <= N - 2; ++i)
            row_simple(src + (i - 1) * N, dst + (i - 1) * N, N);
        return;
    }
    const int64_t K = (J - 8) / 8 + 1; // full 8-wide chunks (J >= 9 here)
    const double *const R0 = src + (r - 1) * N;
    const double *const R1 = src + r * N;
    const double *const R2 = src + (r + 1) * N;
    const double *const R3 = src + (r + 2) * N;
    const double *const R4 = src + (r + 3) * N;
    double *const O1 = dst + r * N;
    double *const O2 = dst + (r + 1) * N;
    double *const O3 = dst + (r + 2) * N;

    __m512d c0 = _mm512_loadu_pd(R0 + 1);
    __m512d p1 = _mm512_set1_pd(R1[0]);
    __m512d c1 = _mm512_loadu_pd(R1 + 1), n1 = _mm512_loadu_pd(R1 + 9);
    __m512d p2 = _mm512_set1_pd(R2[0]);
    __m512d c2 = _mm512_loadu_pd(R2 + 1), n2 = _mm512_loadu_pd(R2 + 9);
    __m512d p3 = _mm512_set1_pd(R3[0]);
    __m512d c3 = _mm512_loadu_pd(R3 + 1), n3 = _mm512_loadu_pd(R3 + 9);
    __m512d c4 = _mm512_loadu_pd(R4 + 1);
    const __m512i iL = mkidxL(), iR = mkidxR();

    for (int64_t k = 0; k < K; ++k) {
        const int64_t j = 1 + 8 * k;
        const __m512d l1 = shiftL(p1, c1, iL), r1 = shiftR(c1, n1, iR);
        _mm512_storeu_pd(O1 + j, out5(c0, l1, c1, r1, c2));
        const __m512d l2 = shiftL(p2, c2, iL), r2 = shiftR(c2, n2, iR);
        _mm512_storeu_pd(O2 + j, out5(c1, l2, c2, r2, c3));
        const __m512d l3 = shiftL(p3, c3, iL), r3 = shiftR(c3, n3, iR);
        _mm512_storeu_pd(O3 + j, out5(c2, l3, c3, r3, c4));
        c0 = _mm512_loadu_pd(R0 + j + 8);
        p1 = c1; c1 = n1; n1 = _mm512_loadu_pd(R1 + j + 16);
        p2 = c2; c2 = n2; n2 = _mm512_loadu_pd(R2 + j + 16);
        p3 = c3; c3 = n3; n3 = _mm512_loadu_pd(R3 + j + 16);
        c4 = _mm512_loadu_pd(R4 + j + 8);
    }
    for (int64_t j = 1 + 8 * K; j <= J; ++j) {
        O1[j] = 0.2 * ((((R1[j] + R1[j - 1]) + R1[j + 1]) + R2[j]) + R0[j]);
        O2[j] = 0.2 * ((((R2[j] + R2[j - 1]) + R2[j + 1]) + R3[j]) + R1[j]);
        O3[j] = 0.2 * ((((R3[j] + R3[j - 1]) + R3[j + 1]) + R4[j]) + R2[j]);
    }
}

static void sweep_fast(const double *restrict src, double *restrict dst,
                       const int64_t N) {
    if (N - 2 < 9) { sweep_simple(src, dst, N, 1, N - 1); return; }
    #pragma omp for schedule(static)
    for (int64_t r = 1; r < N - 1; r += 3)
        window3(src, dst, N, r);
}

static int have_avx512f(void) {
    static int v = -1;
    if (v < 0) v = __builtin_cpu_supports("avx512f") ? 1 : 0;
    return v;
}

// ------------------------------------------------------------------ entry
void jacobi_2d_fp64(double *restrict A, double *restrict B, const int64_t N,
                    const int64_t TSTEPS) {
    if (N < 3 || TSTEPS <= 0) return;
    if (!have_avx512f()) {
        #pragma omp parallel
        {
            for (int64_t t = 0; t < TSTEPS; ++t) {
                #pragma omp for schedule(static)
                for (int64_t i = 1; i < N - 1; ++i)
                    row_simple(A + (i - 1) * N, B + (i - 1) * N, N);
                #pragma omp for schedule(static)
                for (int64_t i = 1; i < N - 1; ++i)
                    row_simple(B + (i - 1) * N, A + (i - 1) * N, N);
            }
        }
        return;
    }
    #pragma omp parallel
    {
        for (int64_t t = 0; t < TSTEPS; ++t) {
            sweep_fast(A, B, N);
            sweep_fast(B, A, N);
        }
    }
}
