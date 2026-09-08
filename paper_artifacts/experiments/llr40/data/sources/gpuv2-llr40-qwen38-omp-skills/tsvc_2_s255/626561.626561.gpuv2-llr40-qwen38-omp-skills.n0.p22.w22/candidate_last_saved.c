#include <stdint.h>
#include <stdio.h>

void tsvc_2_s255_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
    const int64_t n = LEN_1D;
    static int probe = 0;
    if (n <= 0) return;
    if (!probe) { probe = 1; printf("NPROBE n=%lld\n", (long long)n); fflush(stdout); }
    #pragma omp target teams distribute parallel for map(to: b[0:n]) map(from: a[0:n])
    for (int64_t i = 0; i < n; i++) {
        const double m1 = (i >= 1) ? b[i - 1] : b[n - 1];
        const double m2 = (i >= 2) ? b[i - 2] : ((i == 1) ? b[n - 1] : b[n - 2]);
        a[i] = (b[i] + m1 + m2) * 0.333;
    }
}
