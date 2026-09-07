// hpcagent_bench-autogen -- generated from jacobi_2d_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
#include <stdint.h>
#include <stddef.h>

/* One 5-point half-sweep: dst[i][j] = 0.2*(src[i][j]+src[i][j-1]+src[i][j+1]
 * +src[i+1][j]+src[i-1][j]) for interior i,j. Rows are independent, so the
 * row loop is thread-parallel; the j loop walks five lock-step pointer
 * sequences so it auto-vectorizes. */
static inline void sweep(const double *restrict src, double *restrict dst,
                         const int64_t N) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 1; i < N - 1; ++i) {
        const double *u = src + (i - 1) * (size_t)N + 1;
        const double *l = src + (size_t)i * (size_t)N; /* center row, j offset -1 */
        const double *c = l + 1;
        const double *r = l + 2;
        const double *d = src + (i + 1) * (size_t)N + 1;
        double *o = dst + (size_t)i * (size_t)N + 1;
        for (int64_t k = N - 2; k--;) {
            *o = 0.2 * (*c + *l + *r + *d + *u);
            o++; l++; c++; r++; u++; d++;
        }
    }
}

void jacobi_2d_fp64(double *restrict A, double *restrict B, const int64_t N, const int64_t TSTEPS) {
    for (int64_t t = 0; t < TSTEPS; ++t) {
        sweep(A, B, N);
        sweep(B, A, N);
    }
}
