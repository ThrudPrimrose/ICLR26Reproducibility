#include <stddef.h>
#include <stdint.h>
#include <float.h>
#include <omp.h>
#include <stdlib.h>

/*
 * TSVC_2 kernel s3110 – find maximum element in matrix aa and compute checksum.
 *
 * Reference (numpy) implementation:
 *   maxv   = aa[0,0]
 *   xindex = 0
 *   yindex = 0
 *   for i in range(LEN_2D):
 *       for j in range(LEN_2D):
 *           if aa[i,j] > maxv:
 *               maxv = aa[i,j]
 *               xindex = i
 *               yindex = j
 *   chksum = maxv + (double)xindex + (double)yindex
 *   bb[0,0] = chksum
 *
 * The function follows the ABI used by other TSVC kernels:
 *   void tsvc_2_s3110_fp64(const double *restrict aa,
 *                           double *restrict       bb,
 *                           int64_t                LEN_2D,
 *                           uint8_t *restrict      workspace,
 *                           int64_t                workspace_bytes)
 *
 * `workspace` arguments are unused but part of the ABI.
 */

void tsvc_2_s3110_fp64(const double *restrict aa,
                        double *restrict       bb,
                        int64_t                LEN_2D,
                        uint8_t *restrict      workspace,
                        int64_t                workspace_bytes)
{
    (void)workspace;        // Unused, required by the ABI.
    (void)workspace_bytes; // Unused, required by the ABI.
    if (LEN_2D <= 0) {
        // Empty matrix – define checksum as 0.
        bb[0] = 0.0;
        return;
    }

    // Use a parallel reduction: each thread finds a local maximum and its index.
    int64_t nthreads = omp_get_max_threads();
    double *thread_max = (double *)malloc(sizeof(double) * nthreads);
    int64_t *thread_idx = (int64_t *)malloc(sizeof(int64_t) * nthreads);
    if (!thread_max || !thread_idx) {
        // Allocation failure – fall back to a sequential scan.
        double maxv = -DBL_MAX;
        int64_t best_idx = -1; // linear index i*LEN_2D + j
        for (int64_t i = 0; i < LEN_2D; ++i) {
            int64_t row = i * LEN_2D;
            for (int64_t j = 0; j < LEN_2D; ++j) {
                double val = aa[row + j];
                int64_t idx = row + j;
                if (val > maxv) {
                    maxv = val;
                    best_idx = idx;
                }
            }
        }
        int64_t xindex = best_idx / LEN_2D;
        int64_t yindex = best_idx % LEN_2D;
        double chksum = maxv + (double)xindex + (double)yindex;
        bb[0] = chksum;
        if (thread_max) free(thread_max);
        if (thread_idx) free(thread_idx);
        return;
    }

    // Initialise per‑thread results.
    for (int64_t t = 0; t < nthreads; ++t) {
        thread_max[t] = -DBL_MAX;
        thread_idx[t] = -1;
    }

    // Parallel scan over the matrix.
    #pragma omp parallel
    {
        int64_t tid = omp_get_thread_num();
        double local_max = -DBL_MAX;
        int64_t local_idx = -1;
        // Collapse the two nested loops into a single index space for better load‑balance.
        #pragma omp for schedule(static) nowait
        for (int64_t idx = 0; idx < LEN_2D * LEN_2D; ++idx) {
            double val = aa[idx];
            if (val > local_max) {
                local_max = val;
                local_idx = idx;
            }
        }
        thread_max[tid] = local_max;
        thread_idx[tid] = local_idx;
    }

    // Reduce per‑thread results to global maximum.
    double maxv = -DBL_MAX;
    int64_t best_idx = -1;
    for (int64_t t = 0; t < nthreads; ++t) {
        double tv = thread_max[t];
        int64_t ti = thread_idx[t];
        if (tv > maxv) {
            maxv = tv;
            best_idx = ti;
        } else if (tv == maxv && ti >= 0 && (best_idx < 0 || ti < best_idx)) {
            // Tie – choose the smallest linear index (earliest in row‑major order).
            best_idx = ti;
        }
    }

    // Convert linear index back to row/column.
    int64_t xindex = best_idx / LEN_2D;
    int64_t yindex = best_idx % LEN_2D;
    double chksum = maxv + (double)xindex + (double)yindex;
    bb[0] = chksum;

    free(thread_max);
    free(thread_idx);
}
