#include <stdint.h>
#include <omp.h>

void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D, const int64_t S) {
    const double s = (double)S;
    if (LEN_1D <= 0) return;
    if (LEN_1D < 8*1024*1024) {
        for (int64_t i = 0; i < LEN_1D; ++i) a[i] += b[i] * s;
        return;
    }
    const int64_t g = (LEN_1D * 50) / 100;  /* GPU handles [0, g) */
    const int k = 2;                        /* GPU driver threads (parallel copies) */
    #pragma omp parallel
    {
        const int nt = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        if (tid < k) {
            const int64_t lo = (g * tid) / k;
            const int64_t hi = (g * (tid + 1)) / k;
            #pragma omp target teams distribute parallel for simd \
                map(tofrom: a[lo:hi]) map(to: b[lo:hi]) map(to: s)
            for (int64_t i = lo; i < hi; ++i) a[i] += b[i] * s;
        } else {
            const int64_t m = LEN_1D - g;
            const int nh = nt - k;
            const int64_t lo = g + m * (tid - k) / nh;
            const int64_t hi = g + m * (tid - k + 1) / nh;
            for (int64_t i = lo; i < hi; ++i) a[i] += b[i] * s;
        }
    }
}
