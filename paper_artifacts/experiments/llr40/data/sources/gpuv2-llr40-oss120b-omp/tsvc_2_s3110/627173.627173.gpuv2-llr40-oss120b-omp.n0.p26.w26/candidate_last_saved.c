#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>

static inline int64_t idx(int64_t i, int64_t j, int64_t n) { return i * n + j; }

void tsvc_2_s3110_fp64(const double *restrict aa, double *restrict bb, const int64_t LEN_2D) {
    int64_t N = LEN_2D * LEN_2D;
    double maxv = -INFINITY;
    #pragma omp target teams distribute parallel for map(to: aa[0:N]) map(tofrom: maxv) reduction(max:maxv) schedule(static)
    for (int64_t k = 0; k < N; ++k) {
        double v = aa[k];
        if (v > maxv) {
            maxv = v;
        }
    }
    // Find first occurrence of the maximum value on host
    int64_t xindex = 0, yindex = 0;
    bool found = false;
    for (int64_t i = 0; i < LEN_2D && !found; ++i) {
        for (int64_t j = 0; j < LEN_2D; ++j) {
            if (aa[idx(i, j, LEN_2D)] == maxv) {
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
