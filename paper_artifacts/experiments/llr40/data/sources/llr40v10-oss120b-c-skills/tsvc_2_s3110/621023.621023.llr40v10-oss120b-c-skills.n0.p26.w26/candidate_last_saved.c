/* Optimized version of tsvc_2_s3110 using OpenMP parallel reduction
   Finds the maximum element in a LEN_2D x LEN_2D matrix and returns
   max + xindex + yindex as checksum in bb[0]. */

#include <stdint.h>
#include <math.h>
#include <float.h>
#include <omp.h>

/* Compute linear index in row-major order */
static inline int64_t idx(int64_t i, int64_t j, int64_t n) {
    return i * n + j;
}

/* Structure to hold max value and its coordinates */
typedef struct {
    double maxv;
    int64_t x;
    int64_t y;
} max_idx_t;

/* Custom OpenMP reduction to combine max_idx_t values.
   Choose the larger maxv; if equal, pick the smaller (x,y) in row-major order. */
#pragma omp declare reduction (max_idx : max_idx_t : \
    omp_out = (omp_in.maxv > omp_out.maxv) ? omp_in : \
              ((omp_in.maxv < omp_out.maxv) ? omp_out : \
               ((omp_in.x < omp_out.x) || (omp_in.x == omp_out.x && omp_in.y < omp_out.y) ? omp_in : omp_out))) \
    initializer (omp_priv = (max_idx_t){-INFINITY, INT64_MAX, INT64_MAX})

void tsvc_2_s3110_fp64(const double *restrict aa, double *restrict bb, const int64_t LEN_2D) {
    /* Start with the first element as initial maximum */
    max_idx_t global_max = {aa[idx(0, 0, LEN_2D)], 0, 0};

    #pragma omp parallel for schedule(static) reduction(max_idx:global_max)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        max_idx_t local = {-INFINITY, INT64_MAX, INT64_MAX};
        for (int64_t j = 0; j < LEN_2D; ++j) {
            double v = aa[idx(i, j, LEN_2D)];
            if (v > local.maxv || (v == local.maxv && (i < local.x || (i == local.x && j < local.y)))) {
                local.maxv = v;
                local.x = i;
                local.y = j;
            }
        }
        /* Combine the best of this row with the thread's reduction variable */
        global_max = (local.maxv > global_max.maxv) ? local : \
                     ((local.maxv < global_max.maxv) ? global_max : \
                      ((local.x < global_max.x) || (local.x == global_max.x && local.y < global_max.y)) ? local : global_max);
    }

    double chksum = global_max.maxv + (double)global_max.x + (double)global_max.y;
    bb[0] = chksum;
}
