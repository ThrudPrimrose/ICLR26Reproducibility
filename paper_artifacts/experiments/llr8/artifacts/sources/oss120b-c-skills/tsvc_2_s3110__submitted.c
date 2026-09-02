#include <stdint.h>
#include <omp.h>
#include <math.h>

void tsvc_2_s3110_fp64(double *aa, double *bb, int64_t LEN_2D, uint8_t *workspace, int64_t workspace_bytes) {
    double maxv = -INFINITY;
    int64_t xindex = 0;
    int64_t yindex = 0;

    #pragma omp parallel
    {
        double thread_max = -INFINITY;
        int64_t thread_x = 0;
        int64_t thread_y = 0;
        #pragma omp for schedule(static) nowait
        for (int64_t i = 0; i < LEN_2D; ++i) {
            for (int64_t j = 0; j < LEN_2D; ++j) {
                double v = aa[i * LEN_2D + j];
                if (v > thread_max) {
                    thread_max = v;
                    thread_x = i;
                    thread_y = j;
                }
            }
        }
        #pragma omp critical
        {
            if (thread_max > maxv) {
                maxv = thread_max;
                xindex = thread_x;
                yindex = thread_y;
            }
        }
    }

    double chksum = maxv + (double)xindex + (double)yindex;
    bb[0] = chksum;
}
