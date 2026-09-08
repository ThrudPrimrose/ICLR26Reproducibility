#include <stdint.h>
#include <float.h>
#include <omp.h>

#include <stdlib.h>

/*
 * Optimized version of TSVC tsvc_2 kernel s3110 (fp64).
 * Finds the maximum element in a LEN_2D x LEN_2D matrix stored in row-major order,
 * and returns max + row_index + col_index via bb[0].
 * Implements a single-pass parallel reduction with proper tie-breaking to match the
 * reference sequential algorithm (earliest occurrence of the maximum).
 */

struct Best {
    double maxv;
    int64_t idx; // flat index (i * LEN_2D + j)
};

/* Custom OpenMP reduction for struct Best.
 * Chooses the larger maxv; on equal maxv selects the smaller flat index.
 */
/* Removed custom reduction due to compiler bug */

#if 0
void tsvc_2_s3110_fp64(const double *restrict aa, double *restrict bb, const int64_t LEN_2D) {
    const int64_t total = LEN_2D * LEN_2D;
    struct Best best = { -DBL_MAX, INT64_MAX };
    #pragma omp parallel for reduction(maxbest:best) schedule(static)
    for (int64_t flat = 0; flat < total; ++flat) {
        double v = aa[flat];
        // Update the thread‑local best if this element is larger, or equal but earlier.
        if (v > best.maxv || (v == best.maxv && flat < best.idx)) {
            best.maxv = v;
            best.idx = flat;
        }
    }
    // Compute 2‑D indices from flat index.
    int64_t xindex = best.idx / LEN_2D;
    int64_t yindex = best.idx % LEN_2D;
    double chksum = best.maxv + (double)xindex + (double)yindex;
    bb[0] = chksum;
}
#endif


/* Parallel implementation using per‑thread local reduction */
void tsvc_2_s3110_fp64(const double *restrict aa, double *restrict bb, const int64_t LEN_2D) {
    const int64_t total = LEN_2D * LEN_2D;
    int nt = omp_get_max_threads();
    /* Allocate per‑thread best values on the stack using a VLA */
    struct Best thread_best[nt];
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        struct Best local = { -DBL_MAX, INT64_MAX };
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < total; ++i) {
            double v = aa[i];
            if (v > local.maxv || (v == local.maxv && i < local.idx)) {
                local.maxv = v;
                local.idx = i;
            }
        }
        thread_best[tid] = local;
    }
    struct Best best = { -DBL_MAX, INT64_MAX };
    for (int i = 0; i < nt; ++i) {
        struct Best cur = thread_best[i];
        if (cur.maxv > best.maxv || (cur.maxv == best.maxv && cur.idx < best.idx)) {
            best = cur;
        }
    }
    int64_t xindex = best.idx / LEN_2D;
    int64_t yindex = best.idx % LEN_2D;
    double chksum = best.maxv + (double)xindex + (double)yindex;
    bb[0] = chksum;
}
