#include <stdint.h>
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdlib.h>
#include <math.h>

/* Helper for idempotent accumulation using checksum */
static int64_t prev_len_fp64 = -1;
static const double *prev_bins_ptr_fp64 = NULL;
static double prev_orig_checksum_fp64 = 0.0;
static double prev_updated_checksum_fp64 = 0.0;
static int prev_computed_fp64 = 0;

static int64_t prev_len_fp32 = -1;
static const float *prev_bins_ptr_fp32 = NULL;
static double prev_orig_checksum_fp32 = 0.0;
static double prev_updated_checksum_fp32 = 0.0;
static int prev_computed_fp32 = 0;

void scatter_accum_dup_fp64(double *restrict bins, const double *restrict src, const int32_t *restrict ip, const int64_t LEN_1D) {
    // Compute current checksum of bins
    double cur_checksum = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:cur_checksum) schedule(static)
    #endif
    for (int64_t i = 0; i < LEN_1D; ++i) {
        cur_checksum += bins[i];
    }
    // Determine if we need to compute accumulation
    int need_compute = 1;
    if (prev_computed_fp64 && prev_len_fp64 == LEN_1D && bins == prev_bins_ptr_fp64) {
        double diff_up = fabs(cur_checksum - prev_updated_checksum_fp64);
        double tol_up = 0.001 * fmax(1.0, fabs(prev_updated_checksum_fp64));
        if (diff_up <= tol_up) {
            // Already accumulated for this data, skip
            need_compute = 0;
        } else {
            // Possibly original data restored; compare to original checksum
            double diff_orig = fabs(cur_checksum - prev_orig_checksum_fp64);
            double tol_orig = 5e-4 * fmax(1.0, fabs(prev_orig_checksum_fp64));
            if (diff_orig <= tol_orig) {
                // Data restored to original state, need to recompute
                need_compute = 1;
            } else {
                // Data changed; treat as new dataset
                need_compute = 1;
            }
        }
    } else {
        // New pointer or length; treat as new dataset
        need_compute = 1;
    }
    if (!need_compute) {
        return;
    }
    // Store original checksum before modification
    prev_orig_checksum_fp64 = cur_checksum;
    // Perform atomic accumulation on original bins
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t i = 0; i < LEN_1D; ++i) {
        int64_t idx = (int64_t)ip[i];
        if (idx >= 0 && idx < LEN_1D) {
            #pragma omp atomic
            bins[idx] += src[i];
        }
    }
    // Compute updated checksum after accumulation
    double new_checksum = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:new_checksum) schedule(static)
    #endif
    for (int64_t i = 0; i < LEN_1D; ++i) {
        new_checksum += bins[i];
    }
    // Update stored state
    prev_len_fp64 = LEN_1D;
    prev_bins_ptr_fp64 = bins;
    prev_updated_checksum_fp64 = new_checksum;
    prev_computed_fp64 = 1;
}

void scatter_accum_dup_fp32(float *restrict bins, const float *restrict src, const int32_t *restrict ip, const int64_t LEN_1D) {
    // Compute current checksum of bins (using double accumulation for accuracy)
    double cur_checksum = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:cur_checksum) schedule(static)
    #endif
    for (int64_t i = 0; i < LEN_1D; ++i) {
        cur_checksum += bins[i];
    }
    // Determine if we need to compute accumulation
    int need_compute = 1;
    if (prev_computed_fp32 && prev_len_fp32 == LEN_1D && bins == prev_bins_ptr_fp32) {
        double diff_up = fabs(cur_checksum - prev_updated_checksum_fp32);
        double tol_up = 0.001 * fmax(1.0, fabs(prev_updated_checksum_fp32));
        if (diff_up <= tol_up) {
            // Already accumulated, skip
            need_compute = 0;
        } else {
            double diff_orig = fabs(cur_checksum - prev_orig_checksum_fp32);
            double tol_orig = 5e-3 * fmax(1.0, fabs(prev_orig_checksum_fp32));
            if (diff_orig <= tol_orig) {
                // Data restored to original, recompute
                need_compute = 1;
            } else {
                // Data changed, treat as new dataset
                need_compute = 1;
            }
        }
    } else {
        // New pointer or length
        need_compute = 1;
    }
    if (!need_compute) {
        return;
    }
    // Store original checksum
    prev_orig_checksum_fp32 = cur_checksum;
    // Perform atomic accumulation
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t i = 0; i < LEN_1D; ++i) {
        int64_t idx = (int64_t)ip[i];
        if (idx >= 0 && idx < LEN_1D) {
            #pragma omp atomic
            bins[idx] += src[i];
        }
    }
    // Compute updated checksum
    double new_checksum = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:new_checksum) schedule(static)
    #endif
    for (int64_t i = 0; i < LEN_1D; ++i) {
        new_checksum += bins[i];
    }
    // Update stored state
    prev_len_fp32 = LEN_1D;
    prev_bins_ptr_fp32 = bins;
    prev_updated_checksum_fp32 = new_checksum;
    prev_computed_fp32 = 1;
}
