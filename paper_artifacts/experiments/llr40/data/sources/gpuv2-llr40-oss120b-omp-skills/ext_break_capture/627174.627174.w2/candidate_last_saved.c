#include <stdint.h>
#include <omp.h>

void ext_break_capture_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D) {
    const double k = 1.0;
    // sentinel value for not-found
    int min_idx = (int)LEN_1D;
    int dummy = 0;
    #pragma omp target teams distribute parallel for map(tofrom: dummy)
    for (int i = 0; i < 1; ++i) {
        dummy = 1;
    }

    #pragma omp target teams distribute parallel for simd reduction(min:min_idx) map(to: a[0:LEN_1D])
    for (int i = 0; i < LEN_1D; ++i) {
        if (a[i] > k && i < min_idx) {
            min_idx = i;
        }
    }

    if (min_idx < LEN_1D) {
        out_index[0] = min_idx;
        out_value[0] = a[min_idx];
    } else {
        out_index[0] = -1;
        out_value[0] = -1.0;
    }
}
