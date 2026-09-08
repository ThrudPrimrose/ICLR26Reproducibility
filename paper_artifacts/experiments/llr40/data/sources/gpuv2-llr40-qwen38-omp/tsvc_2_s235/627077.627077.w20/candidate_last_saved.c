/* TSVC s235 fp64 -- OpenMP target offload (AMD MI300A, explicit memory).
 *
 * Reference:
 *   for i: a[i] += b[i]*c[i]
 *          for j=1..N-1: aa[j,i] = aa[j-1,i] + bb[j,i]*a[i]
 *
 * The inner loop is a serial scan along j (rows) for each column i.  In the
 * C-major layout aa[j,i]/bb[j,i] are strided by N, so a naive per-column scan
 * is latency-bound.  We parallelize over columns i (independent scans) and give
 * each work-item a block of B strided loads in flight (memory-level
 * parallelism) to hide the serial FADD chain.
 */
#include <stdint.h>

#define S235_BLOCK 8

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa, double *restrict b,
                      const double *restrict bb, const double *restrict c,
                      const int64_t LEN_2D)
{
    const int64_t N = LEN_2D;
    const int64_t B = S235_BLOCK;
    #pragma omp target teams distribute parallel for \
        map(tofrom: a[0:N], aa[0:N*N]) map(to: b[0:N], bb[0:N*N], c[0:N])
    for (int64_t i = 0; i < N; ++i) {
        const double Ai = a[i] + b[i]*c[i];
        a[i] = Ai;
        double run = aa[i];   /* aa[0*N + i] */
        int64_t j = 1;
        for (; j + B <= N; j += B) {
            double bbv[B];
            #pragma omp simd
            for (int k = 0; k < B; ++k) bbv[k] = bb[(j+k)*N + i];
            for (int k = 0; k < B; ++k) { run += bbv[k]*Ai; aa[(j+k)*N + i] = run; }
        }
        for (; j < N; ++j) { run += bb[j*N + i]*Ai; aa[j*N + i] = run; }
    }
}
