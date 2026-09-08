#include <stdint.h>

void wf_triangular_fp64(double *restrict a, const int64_t LEN_2D) {
    const int64_t N = LEN_2D;
    for (int64_t i = 1; i < N; ++i) {
        double *restrict row = a + i * N;
        const double *restrict prev = a + (i - 1) * N;
        double s = row[i - 1];
        for (int64_t j = i; j < N; ++j) {
            row[j] += prev[j];
            s += row[j];
            row[j] = s;
        }
    }
}
