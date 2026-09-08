#include <stdint.h>
#include <omp.h>

void fuse_diamond_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    if (LEN_1D >= 1000000 && LEN_1D <= 103444714) {
        #pragma omp target map(to: a[0:LEN_1D]) map(from: out[0:LEN_1D])
        {
            #pragma omp teams distribute parallel for
            for (int64_t i = 0; i < LEN_1D; ++i) {
                double t = a[i] * a[i];
                out[i] = (t + 1.0) * (t - 1.0);
            }
        }
    } else if (LEN_1D <= 65536) {
        for (int64_t i = 0; i < LEN_1D; ++i) {
            double t = a[i] * a[i];
            out[i] = (t + 1.0) * (t - 1.0);
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            double t = a[i] * a[i];
            out[i] = (t + 1.0) * (t - 1.0);
        }
    }
}
