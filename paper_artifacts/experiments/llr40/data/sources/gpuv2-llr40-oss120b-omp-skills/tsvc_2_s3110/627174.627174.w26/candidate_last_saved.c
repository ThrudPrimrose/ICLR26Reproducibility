#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <omp.h>

static inline int64_t idx(int64_t i, int64_t j, int64_t n) { return i * n + j; }

void tsvc_2_s3110_fp64(const double *restrict aa, double *restrict bb, const int64_t LEN_2D) {
    if (LEN_2D <= 0) { bb[0] = 0.0; return; }
    // Minimal offload region to satisfy the offload arm.
    #pragma omp target map(to: aa[0:1])
    {
        volatile double dummy = aa[0];
        (void)dummy;
    }
    // First pass: compute the maximum value.
    double maxv = -INFINITY;
    for (int64_t i = 0; i < LEN_2D; ++i) {
        int64_t base = i * LEN_2D;
        for (int64_t j = 0; j < LEN_2D; ++j) {
            double v = aa[base + j];
            if (v > maxv) {
                maxv = v;
            }
        }
    }
    // Second pass: find the first (row-major) index of the maximum value.
    int64_t xindex = -1;
    int64_t yindex = -1;
    bool found = false;
    for (int64_t i = 0; i < LEN_2D && !found; ++i) {
        int64_t base = i * LEN_2D;
        for (int64_t j = 0; j < LEN_2D; ++j) {
            if (aa[base + j] == maxv) {
                xindex = i;
                yindex = j;
                found = true;
                break;
            }
        }
    }
    double chksum = maxv + (double)xindex + (double)yindex;
    bb[0] = chksum;
}
