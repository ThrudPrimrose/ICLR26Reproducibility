/* Optimized argmax_with_index kernel for fp64.
 * Implements argmax_with_index_fp64(const double *restrict a,
 *                                  int64_t *restrict out_index,
 *                                  double *restrict out_value,
 *                                  const int64_t LEN_1D)
 *
 * This version uses AVX2 intrinsics to vectorize the reduction and
 * OpenMP for multi‑threading when many elements are present. It falls back to
 * scalar code on systems without AVX2.
 *
 * The algorithm maintains per‑lane maximum values and their indices in SIMD
 * registers, updates them with a blend based on a comparison mask, and finally
 * reduces the SIMD lanes to a scalar maximum.
 *
 * For large arrays the work is divided among threads using OpenMP. Each thread
 * processes a contiguous sub‑range, performing the same vectorized reduction.
 * After the parallel region the per‑thread results are combined sequentially,
 * preserving the rule that the first occurrence of the maximum is returned
 * (i.e. ties keep the lower index).
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <math.h>

/* Helper struct for per‑thread reduction results. */
typedef struct {
    double val;   // maximum value
    int64_t idx;  // index of the maximum (0‑based)
} maxpair_t;

/* Combine two maxpair_t values, keeping the first index on ties. */
static inline maxpair_t maxpair_combine(maxpair_t a, maxpair_t b) {
    if (b.val > a.val) {
        return b;
    } else if (b.val < a.val) {
        return a;
    } else {
        // values are equal – keep the smaller index (earliest occurrence)
        return (b.idx < a.idx) ? b : a;
    }
}

/* Scalar fallback for a sub‑range [start, end). */
static inline maxpair_t scalar_maxpair(const double *restrict a, int64_t start, int64_t end) {
    maxpair_t r = {a[start], start};
    for (int64_t i = start + 1; i < end; ++i) {
        double v = a[i];
        if (v > r.val) {
            r.val = v;
            r.idx = i;
        }
    }
    return r;
}

/* Vectorized reduction over a sub‑range [start, end) (end exclusive).
 * Returns a maxpair_t with the maximum value and its index.
 */
static inline maxpair_t vectorized_maxpair(const double *restrict a, int64_t start, int64_t end) {
#if defined(__AVX2__)
    const int64_t vec_width = 4; // AVX2 processes 4 doubles per 256‑bit register
    int64_t len = end - start;
    if (len <= vec_width) {
        // Small range – use scalar loop
        return scalar_maxpair(a, start, end);
    }

    // Initialize SIMD registers with the first element broadcast
    __m256d v_max = _mm256_set1_pd(a[start]);
    __m256i v_idx = _mm256_set_epi64x(start + 3, start + 2, start + 1, start);

    int64_t i = start;
    for (; i + vec_width <= end; i += vec_width) {
        __m256d v = _mm256_loadu_pd(&a[i]);
        __m256d mask = _mm256_cmp_pd(v, v_max, _CMP_GT_OQ);
        v_max = _mm256_blendv_pd(v_max, v, mask);
        __m256i cur_idx = _mm256_set_epi64x(i + 3, i + 2, i + 1, i);
        v_idx = _mm256_blendv_epi8(v_idx, cur_idx, _mm256_castpd_si256(mask));
    }

    // Reduce SIMD lanes to a scalar result
    double lane_vals[4];
    int64_t lane_idxs[4];
    _mm256_storeu_pd(lane_vals, v_max);
    _mm256_storeu_si256((__m256i *)lane_idxs, v_idx);
    maxpair_t best = {lane_vals[0], lane_idxs[0]};
    for (int l = 1; l < 4; ++l) {
        if (lane_vals[l] > best.val) {
            best.val = lane_vals[l];
            best.idx = lane_idxs[l];
        } else if (lane_vals[l] == best.val && lane_idxs[l] < best.idx) {
            best.idx = lane_idxs[l];
        }
    }

    // Tail elements that do not fill a full vector
    for (; i < end; ++i) {
        double v = a[i];
        if (v > best.val) {
            best.val = v;
            best.idx = i;
        }
    }
    return best;
#else
    // No AVX2 – fall back to scalar implementation
    return scalar_maxpair(a, start, end);
#endif
}

/* Main kernel entry point as required by the benchmark harness. */
void argmax_with_index_fp64(const double *restrict a,
                            int64_t *restrict out_index,
                            double *restrict out_value,
                            const int64_t LEN_1D) {
    // The reference assumes LEN_1D >= 1.
    if (LEN_1D <= 0) {
        out_value[0] = NAN; // NaN signals invalid input
        out_index[0] = -1;
        return;
    }

    maxpair_t global;
#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
    if (max_threads > LEN_1D) max_threads = (int)LEN_1D; // do not spawn more threads than elements
    maxpair_t *local = (maxpair_t *)malloc(max_threads * sizeof(maxpair_t));
    #pragma omp parallel num_threads(max_threads)
    {
        int tid = omp_get_thread_num();
        int64_t chunk = LEN_1D / max_threads;
        int64_t rem = LEN_1D % max_threads;
        int64_t start = tid * chunk + (tid < rem ? tid : rem);
        int64_t end = start + chunk + (tid < rem ? 1 : 0);
        // Each thread computes its local maximum over its sub‑range.
        local[tid] = vectorized_maxpair(a, start, end);
    }
    // Reduce across threads (sequential, preserves first‑occurrence on ties).
    global = local[0];
    for (int t = 1; t < max_threads; ++t) {
        global = maxpair_combine(global, local[t]);
    }
    free(local);
#else
    // Single‑threaded execution.
    global = vectorized_maxpair(a, 0, LEN_1D);
#endif

    out_value[0] = global.val;
    out_index[0] = global.idx;
}

