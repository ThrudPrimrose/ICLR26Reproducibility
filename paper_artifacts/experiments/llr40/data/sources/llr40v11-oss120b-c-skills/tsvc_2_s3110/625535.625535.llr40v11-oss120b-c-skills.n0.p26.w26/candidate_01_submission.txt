/* Optimized version of tsvc_2_s3110_fp64: parallel max reduction with indices.
   The reference implementation scans a LEN_2D x LEN_2D matrix 'aa' to find the maximum value
   and its indices, then writes a checksum into bb[0] = max + xindex + yindex.
*/

#include <stdint.h>
#include <math.h>
#include <omp.h>

static inline int64_t idx(int64_t i, int64_t j, int64_t n) { return i * n + j; }

void tsvc_2_s3110_fp64(const double *restrict aa, double *restrict bb, const int64_t LEN_2D) {
    double maxv = -INFINITY;
    int64_t xindex = -1;
    int64_t yindex = -1;

    #pragma omp parallel
    {
        double maxv_loc = -INFINITY;
        int64_t xi_loc = -1;
        int64_t yj_loc = -1;

        #pragma omp for schedule(static) nowait
        for (int64_t i = 0; i < LEN_2D; ++i) {
            const double *row = aa + idx(i, 0, LEN_2D);
            for (int64_t j = 0; j < LEN_2D; ++j) {
                double v = row[j];
                if (v > maxv_loc) {
                    maxv_loc = v;
                    xi_loc = i;
                    yj_loc = j;
                }
            }
        }
        #pragma omp critical
        {
            if (maxv_loc > maxv) {
                maxv = maxv_loc;
                xindex = xi_loc;
                yindex = yj_loc;
            }
        }
    }
    double chksum = maxv + (double)xindex + (double)yindex;
    bb[0] = chksum;
}

