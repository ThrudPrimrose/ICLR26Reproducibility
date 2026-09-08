#include <stdint.h>
#include <omp.h>

void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    const int64_t n = LEN_2D;
    if (n <= 1) return;
    const int64_t nn = n * n;
    const int64_t ndiag = 2 * n - 3;

    #pragma omp target teams distribute parallel for \
        map(to: bb[0:nn]) map(tofrom: aa[0:nn])
    for (int64_t cp = 0; cp < ndiag; ++cp) {
        int64_t c = cp + 2 - n;             // diagonal i - j == c
        int64_t j0 = (1 - c > 1) ? (1 - c) : 1;
        int64_t j1 = (n - 1 - c < n - 1) ? (n - 1 - c) : (n - 1);
        int64_t i0 = j0 + c;
        double acc = aa[(i0 - 1) * n + (j0 - 1)];
        for (int64_t j = j0; j <= j1; ++j) {
            int64_t i = j + c;
            acc += bb[i * n + j];
            aa[i * n + j] = acc;
        }
    }
}
