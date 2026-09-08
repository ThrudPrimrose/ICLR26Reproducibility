#include <math.h>
#include <stdint.h>

void tsvc_2_s318_fp64(const double *restrict a, double *restrict result, const int64_t LEN_1D, const int64_t inc) {
    double maxv = -1.0;
    int64_t index = 0;

    #pragma omp parallel
    {
        double local_maxv = -1.0;
        int64_t local_index = 0;

        #pragma omp for nowait
        for (int64_t i = 0; i < LEN_1D; ++i) {
            double v = fabs(a[i * inc]);
            if (v > local_maxv) {
                local_maxv = v;
                local_index = i;
            }
        }

        #pragma omp critical
        {
            if (local_maxv > maxv || (local_maxv == maxv && local_index < index)) {
                maxv = local_maxv;
                index = local_index;
            }
        }
    }

    double chksum = maxv + (double)(index);
    result[0] = chksum;
}
