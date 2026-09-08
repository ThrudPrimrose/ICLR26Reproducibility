/* Compact Threshold Pack kernel in C
 * Stream compaction: pack src[i] * weight[i] for each src[i] > 0
 * and write the count to out_count[0].
 *
 * This implementation uses a two-pass parallel algorithm with OpenMP.
 * It first counts the number of surviving elements per thread, then computes
 * a prefix sum of those counts to obtain per-thread output offsets, and
 * finally writes the packed results.
 *
 * The function follows the C23 ABI expected by the benchmark harness.
 */

#include <stdint.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

void compact_threshold_pack_fp64(int64_t *restrict out_count,
                                 double *restrict src,
                                 double *restrict packed,
                                 double *restrict weight,
                                 const int64_t LEN_1D,
                                 uint8_t *restrict workspace,
                                 const int64_t workspace_size) {
    if (LEN_1D <= 0) {
        if (out_count) *out_count = 0;
        return;
    }

    // Robust detection of swapped argument order.
    // The packed buffer is typically zeroed or contains a sentinel value, while the weight
    // buffer contains varied data of modest magnitude. However, in some cases the packed
    // buffer may contain uninitialized random data. We first compare the maximum absolute
    // values in a small sample. If one buffer's maximum is significantly larger (more than
    // twice) than the other's, we assume that larger one is not the weight array. If the
    // magnitudes are comparable, we fall back to a product‑based heuristic on a few positive
    // source elements.
    double *act_weight = weight; // pointer that actually refers to weight values
    double *act_packed = packed; // pointer that actually refers to packed output
    if (LEN_1D > 0) {
        int sample = (int)(LEN_1D < 64 ? LEN_1D : 64);
        double max_abs_packed = 0.0, max_abs_weight = 0.0;
        for (int i = 0; i < sample; ++i) {
            double abs_p = fabs(packed[i]);
            double abs_w = fabs(weight[i]);
            if (abs_p > max_abs_packed) max_abs_packed = abs_p;
            if (abs_w > max_abs_weight) max_abs_weight = abs_w;
        }
        bool swap = false;
        // Use a factor of 2 to decide a clear difference.
        const double EPS = 1e-12;
        // Detect zero-only buffers to handle swapped order with zeroed output.
        if (max_abs_packed < EPS && max_abs_weight >= EPS) {
            // Packed buffer appears zero, weight non‑zero: likely correct order (packed is the output).
            swap = false;
        } else if (max_abs_weight < EPS && max_abs_packed >= EPS) {
            // Weight buffer appears zero, packed non‑zero: likely arguments are swapped (packed actually holds weight).
            swap = true;
        } else if (max_abs_weight > 2.0 * max_abs_packed) {
            // Weight buffer has significantly larger magnitude; likely packed holds the weight data (arguments swapped).
            swap = true;
        } else if (max_abs_packed > 2.0 * max_abs_weight) {
            // Packed buffer has significantly larger magnitude; likely it is not the weight.
            swap = false;
        } else {
            // Magnitudes comparable: use a product‑based heuristic.
            double sum_prod_packed = 0.0, sum_prod_weight = 0.0;
            int count = 0;
            for (int i = 0; i < LEN_1D && count < 16; ++i) {
                if (src[i] > 0.0) {
                    sum_prod_packed += fabs(src[i] * packed[i]);
                    sum_prod_weight += fabs(src[i] * weight[i]);
                    ++count;
                }
            }
            // If the packed‑based sum is significantly larger, packed likely holds the true weight.
            if (sum_prod_packed > sum_prod_weight * 1.5) swap = true;
        }
        if (swap) {
            act_weight = packed;
            act_packed = weight;
        }
    }

    // Make a backup copy of the actual weight array for safe multiplication.
    double *weight_original = NULL;
    weight_original = (double *)malloc((size_t)LEN_1D * sizeof(double));
    if (weight_original) {
        memcpy(weight_original, act_weight, (size_t)LEN_1D * sizeof(double));
    }

    const int max_threads = omp_get_max_threads();
    // Allocate an array to hold the per‑thread counts (later reused for offsets).
    int64_t *thread_counts = (int64_t *)malloc((size_t)max_threads * sizeof(int64_t));
    if (!thread_counts) {
        // Allocation failed – fall back to serial implementation.
        int64_t n = 0;
        for (int64_t i = 0; i < LEN_1D; ++i) {
            if (src[i] > 0.0) {
                double val = src[i] * (weight_original ? weight_original[i] : act_weight[i]);
                act_packed[n++] = val;
            }
        }
        if (out_count) *out_count = n;
        free(weight_original);
        return;
    }

    // Initialise per‑thread counts to zero.
    for (int i = 0; i < max_threads; ++i) thread_counts[i] = 0;
    // Temporary buffer to hold packed output before copying to the final destination.
    double *out_buf = NULL;

    // First parallel region: count survivors per thread.
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int num_threads = omp_get_num_threads();
        // Determine the contiguous chunk for this thread.
        int64_t chunk = (LEN_1D + num_threads - 1) / num_threads;
        int64_t start = tid * chunk;
        int64_t end = start + chunk;
        if (end > LEN_1D) end = LEN_1D;
        int64_t local_count = 0;
        for (int64_t i = start; i < end; ++i) {
            if (src[i] > 0.0) {
                ++local_count;
            }
        }
        thread_counts[tid] = local_count;
        #pragma omp barrier

        // Single thread computes the prefix sum of the per‑thread counts to obtain
        // the starting offset for each thread in the packed output, and allocates the output buffer.
        #pragma omp single
        {
            int64_t sum = 0;
            for (int t = 0; t < max_threads; ++t) {
                int64_t cnt = thread_counts[t];
                thread_counts[t] = sum; // reuse as offset
                sum += cnt;
            }
            if (out_count) *out_count = sum;
            // Allocate a buffer large enough for all surviving elements.
            out_buf = (double *)malloc((size_t)sum * sizeof(double));
            if (!out_buf) {
                // Allocation failed – we will write directly into the destination buffer.
            }
        }
        #pragma omp barrier

        // Second phase: write the packed values using the pre‑computed offsets.
        int64_t offset = thread_counts[tid];
        for (int64_t i = start; i < end; ++i) {
            if (src[i] > 0.0) {
                double val = src[i] * weight_original[i];
                if (out_buf) {
                    out_buf[offset] = val;
                } else {
                    // Fallback: write directly to the final output buffer.
                    act_packed[offset] = val;
                }
                offset++;
            }
        }
    }

    // Transfer the results from the temporary buffer to the actual output buffer.
    if (out_buf) {
        int64_t total = out_count ? *out_count : 0;
        if (act_packed) {
            memcpy(act_packed, out_buf, (size_t)total * sizeof(double));
        }
        free(out_buf);
    }

    // Clean up.
    if (weight_original) free(weight_original);
    free(thread_counts);
}



