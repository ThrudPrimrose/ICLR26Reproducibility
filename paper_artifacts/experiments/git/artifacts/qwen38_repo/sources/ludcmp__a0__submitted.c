/* hpcagent_bench-autogen -- generated from ludcmp_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override. */
/* Optimized right-looking Doolittle LU without pivoting + forward/back substitution.
 *
 * Bit-identical factorization to the numpy reference: per k step the column below the
 * diagonal is scaled by true division and the rank-1 trailing update is applied
 * elementwise as A[i][j] -= A[i][k] * A[k][j] with separate mul/sub roundings (the
 * fp-contract=off attribute keeps the compiler from contracting them into an FMA,
 * which numpy does not do).  The whole k loop runs inside one persistent OpenMP
 * team; each k step is a static-schedule `omp for` over rows (one implicit barrier),
 * and the short tail is done serially so the last ~100 steps do not pay team
 * overhead.  The two triangular solves keep the reference's sequential recurrence.
 */
#include <stdint.h>
#include <math.h>
#include <omp.h>

__attribute__((optimize("fp-contract=off")))
static void factor_rows(double *restrict A, const int64_t k, const int64_t N,
                        const int64_t i_first, const int64_t i_last) {
    const double Akk = A[k * N + k];
    const double *restrict rowk = A + k * N;
    for (int64_t i = i_first; i < i_last; ++i) {
        double *restrict rowi = A + i * N;
        const double aik = rowi[k] / Akk;
        rowi[k] = aik;
        for (int64_t j = k + 1; j < N; ++j) {
            double p = aik * rowk[j];
            rowi[j] -= p;
        }
    }
}

__attribute__((optimize("fp-contract=off")))
static void factor_parallel(double *restrict A, const int64_t N, const int64_t k_first, const int64_t k_last) {
    #pragma omp parallel
    for (int64_t k = k_first; k < k_last; ++k) {
        #pragma omp for schedule(static)
        for (int64_t i = k + 1; i < N; ++i) {
            double *restrict rowi = A + i * N;
            const double aik = rowi[k] / A[k * N + k];
            rowi[k] = aik;
            const double *restrict rowk = A + k * N;
            for (int64_t j = k + 1; j < N; ++j) {
                double p = aik * rowk[j];
                rowi[j] -= p;
            }
        }
    }
}

__attribute__((optimize("fp-contract=off")))
static void solve_pack(double *restrict A, const double *restrict b, double *restrict x, double *restrict y, const int64_t N) {
    for (int64_t i = 0; i < N; ++i) {
        const double *restrict rowi = A + i * N;
        double s = 0.0;
        for (int64_t j = 0; j < i; ++j)
            s += rowi[j] * y[j];
        y[i] = b[i] - s;
    }
    for (int64_t i = N - 1; i >= 0; --i) {
        const double *restrict rowi = A + i * N;
        double s = 0.0;
        for (int64_t j = i + 1; j < N; ++j)
            s += rowi[j] * x[j];
        x[i] = (y[i] - s) / rowi[i];
    }
}

void ludcmp_fp64(double *restrict A, const double *restrict b, double *restrict x, double *restrict y, const int64_t N) {
    const int64_t k_split = N - 96;  /* last 96 steps: serial (parallel overhead > work) */
    if (N >= 256) {
        int mt = omp_get_max_threads();
        if (mt > 8) mt = 8;
        if (mt >= 2 && k_split > 0) {
            factor_parallel(A, N, 0, k_split);
            for (int64_t k = k_split; k < N; ++k)
                factor_rows(A, k, N, k + 1, N);
            solve_pack(A, b, x, y, N);
            return;
        }
    }
    for (int64_t k = 0; k < N; ++k)
        factor_rows(A, k, N, k + 1, N);
    solve_pack(A, b, x, y, N);
}
