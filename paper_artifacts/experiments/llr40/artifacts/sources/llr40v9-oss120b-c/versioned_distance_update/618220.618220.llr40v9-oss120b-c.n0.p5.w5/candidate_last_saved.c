#include <stdint.h>
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdlib.h>

/*
 * versioned_distance_update kernel
 *
 * Compute a[i] = 0.75 * a[i - K] + b[i] * c[i] for i = K .. LEN_1D-1.
 * The first K elements of a remain unchanged (seed values).
 *
 * This implementation parallelises across the K independent chains using OpenMP.
 * For K == 1 the recurrence is a simple serial scan, so we handle it with a plain loop.
 */

void versioned_distance_update_fp64(double *restrict a,
                                   const double *restrict b,
                                   const double *restrict c,
                                   const int64_t K,
                                   const int64_t LEN_1D) {
    if (LEN_1D <= K || K <= 0) return; // nothing to do or invalid K
    if (K == 1) {
        // Use a blocked parallel scan for the K=1 case.
        // This parallel algorithm computes the recurrence a[i] = 0.75*a[i-1] + b[i]*c[i]
        // using per‑thread affine transforms (A, B) where a_out = A*a_in + B.
        // If the problem size is small, fall back to a simple serial loop.
        int max_threads = omp_get_max_threads();
        if (LEN_1D < 4 * max_threads) {
            // Small case: just run serial loop.
            for (int64_t i = 1; i < LEN_1D; ++i) {
                a[i] = 0.75 * a[i - 1] + b[i] * c[i];
            }
            return;
        }
        // Allocate per‑thread block transforms.
        typedef struct {
            double A; // multiplicative factor (0.75^len)
            double B; // additive factor
        } Transform;
        Transform *block_t = (Transform *)malloc(max_threads * sizeof(Transform));
        Transform *prefix_t = (Transform *)malloc(max_threads * sizeof(Transform));
        if (!block_t || !prefix_t) {
            // Allocation failed – fall back to serial.
            for (int64_t i = 1; i < LEN_1D; ++i) {
                a[i] = 0.75 * a[i - 1] + b[i] * c[i];
            }
            if (block_t) free(block_t);
            if (prefix_t) free(prefix_t);
            return;
        }
        // ------- First pass: compute block transforms (A,B) -------
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            // Chunk size for this thread (ceil division).
            int64_t chunk = (LEN_1D + nt - 1) / nt;
            int64_t start = (int64_t)tid * chunk;
            int64_t end = start + chunk;
            if (end > LEN_1D) end = LEN_1D;
            // Initialise transform for this block.
            Transform tr;
            tr.A = 1.0;
            tr.B = 0.0;
            // The recurrence starts at i = 1, so skip i = 0.
            int64_t i_start = (start == 0) ? 1 : start;
            for (int64_t i = i_start; i < end; ++i) {
                tr.B = 0.75 * tr.B + b[i] * c[i];
                tr.A *= 0.75;
            }
            block_t[tid] = tr;
        } // implicit barrier
        // ------- Compute prefix transforms of blocks (sequential) -------
        prefix_t[0].A = 1.0;   // identity
        prefix_t[0].B = 0.0;
        for (int t = 1; t < max_threads; ++t) {
            Transform prev = prefix_t[t - 1];
            Transform blk = block_t[t - 1];
            Transform cur;
            cur.A = blk.A * prev.A;
            cur.B = blk.A * prev.B + blk.B;
            prefix_t[t] = cur;
        }
        // ------- Second pass: write out using the proper initial value for each block -------
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            int64_t chunk = (LEN_1D + nt - 1) / nt;
            int64_t start = (int64_t)tid * chunk;
            int64_t end = start + chunk;
            if (end > LEN_1D) end = LEN_1D;
            double a_before;
            if (start == 0) {
                // First block: a[0] already set.
                a_before = a[0];
                // Process from i = 1 onward within this block.
                int64_t i_begin = 1;
                for (int64_t i = i_begin; i < end; ++i) {
                    a[i] = 0.75 * a_before + b[i] * c[i];
                    a_before = a[i];
                }
            } else {
                // Compute a_before = value of a at index start-1 using prefix transform.
                Transform pre = prefix_t[tid];
                double a0 = a[0];
                a_before = pre.A * a0 + pre.B;
                for (int64_t i = start; i < end; ++i) {
                    a[i] = 0.75 * a_before + b[i] * c[i];
                    a_before = a[i];
                }
            }
        }
        free(block_t);
        free(prefix_t);
        return;
    }
    // Parallel over the K independent chains.
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < K; ++s) {
        // Start at the first index >= K belonging to this chain.
        for (int64_t i = s + K; i < LEN_1D; i += K) {
            a[i] = 0.75 * a[i - K] + b[i] * c[i];
        }
    }
}

void versioned_distance_update_fp32(float *restrict a,
                                   const float *restrict b,
                                   const float *restrict c,
                                   const int64_t K,
                                   const int64_t LEN_1D) {
    if (LEN_1D <= K || K <= 0) return;
    if (K == 1) {
        // Blocked parallel scan for K=1 (float version).
        int max_threads = omp_get_max_threads();
        if (LEN_1D < 4 * max_threads) {
            // Small case: serial loop.
            for (int64_t i = 1; i < LEN_1D; ++i) {
                a[i] = 0.75f * a[i - 1] + b[i] * c[i];
            }
            return;
        }
        typedef struct {
            float A;
            float B;
        } TransformF;
        TransformF *block_t = (TransformF *)malloc(max_threads * sizeof(TransformF));
        TransformF *prefix_t = (TransformF *)malloc(max_threads * sizeof(TransformF));
        if (!block_t || !prefix_t) {
            for (int64_t i = 1; i < LEN_1D; ++i) {
                a[i] = 0.75f * a[i - 1] + b[i] * c[i];
            }
            if (block_t) free(block_t);
            if (prefix_t) free(prefix_t);
            return;
        }
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            int64_t chunk = (LEN_1D + nt - 1) / nt;
            int64_t start = (int64_t)tid * chunk;
            int64_t end = start + chunk;
            if (end > LEN_1D) end = LEN_1D;
            TransformF tr;
            tr.A = 1.0f;
            tr.B = 0.0f;
            int64_t i_start = (start == 0) ? 1 : start;
            for (int64_t i = i_start; i < end; ++i) {
                tr.B = 0.75f * tr.B + b[i] * c[i];
                tr.A *= 0.75f;
            }
            block_t[tid] = tr;
        }
        // Compute prefix transforms.
        prefix_t[0].A = 1.0f;
        prefix_t[0].B = 0.0f;
        for (int t = 1; t < max_threads; ++t) {
            TransformF prev = prefix_t[t - 1];
            TransformF blk = block_t[t - 1];
            TransformF cur;
            cur.A = blk.A * prev.A;
            cur.B = blk.A * prev.B + blk.B;
            prefix_t[t] = cur;
        }
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            int64_t chunk = (LEN_1D + nt - 1) / nt;
            int64_t start = (int64_t)tid * chunk;
            int64_t end = start + chunk;
            if (end > LEN_1D) end = LEN_1D;
            float a_before;
            if (start == 0) {
                a_before = a[0];
                for (int64_t i = 1; i < end; ++i) {
                    a[i] = 0.75f * a_before + b[i] * c[i];
                    a_before = a[i];
                }
            } else {
                TransformF pre = prefix_t[tid];
                float a0 = a[0];
                a_before = pre.A * a0 + pre.B;
                for (int64_t i = start; i < end; ++i) {
                    a[i] = 0.75f * a_before + b[i] * c[i];
                    a_before = a[i];
                }
            }
        }
        free(block_t);
        free(prefix_t);
        return;
    }
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < K; ++s) {
        for (int64_t i = s + K; i < LEN_1D; i += K) {
            a[i] = 0.75f * a[i - K] + b[i] * c[i];
        }
    }
}
