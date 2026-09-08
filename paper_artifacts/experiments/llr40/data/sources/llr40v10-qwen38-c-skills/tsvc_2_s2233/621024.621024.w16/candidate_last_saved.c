/* tsvc_2_s2233 optimized.
 *
 * Reference semantics (re-derived from /shared/tasks/tsvc_2_s2233/):
 *   for i in 8..N-1: for j in 8..N-1: aa[j*N+i] = aa[(j-1)*N+i] + cc[j*N+i]
 *   for i in 8..N-1: for j in 8..N-1: bb[i*N+j] = bb[(i-1)*N+j] + cc[i*N+j]
 * i.e. for every column k in 8..N-1 a running sum down the column, base at
 * row 7 (read, never written), into aa and bb respectively.
 *
 * The two loops write disjoint arrays -> fission is legal. The dependence is
 * the running sum, which is carried over ROWS; columns are independent.
 * So: parallel over column spans; each thread scans all rows for its span,
 * keeping the per-column running sums (its state) in L1-hot local arrays.
 * Per-lane FP order matches the reference exactly (base first, then +cc in
 * increasing row order) -> bit-identical results.
 *
 * Memory: each thread's per-row span is a wide contiguous run, so every row
 * is streamed with unit stride at full vector width; total DRAM traffic is
 * the minimum 3*(N-8)^2 doubles (cc read once, aa/bb written once).
 *
 * Small N: the OpenMP region costs more than the work; run serial.
 */
#include <stdint.h>
#include <omp.h>

static void span_work(double *restrict aa, double *restrict bb, const double *restrict cc,
                      const int64_t N, const int64_t K0, const int64_t W) {
  double sa[W];
  double sb[W];
  const double *ra = aa + 7 * N + K0;
  const double *rb = bb + 7 * N + K0;
  for (int64_t m = 0; m < W; ++m) {
    sa[m] = ra[m];
    sb[m] = rb[m];
  }
  for (int64_t j = 8; j < N; ++j) {
    const double *ccr = cc + j * N + K0;
    double *aar = aa + j * N + K0;
    double *bbr = bb + j * N + K0;
    for (int64_t m = 0; m < W; ++m) {
      double c = ccr[m];
      sa[m] += c;
      sb[m] += c;
      aar[m] = sa[m];
      bbr[m] = sb[m];
    }
  }
}

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N <= 8) return;
  const int64_t ncol = N - 8;

  if (ncol < 128) {
    span_work(aa, bb, cc, N, 8, ncol);
    return;
  }

  #pragma omp parallel
  {
    const int64_t t = omp_get_thread_num();
    const int64_t nt = omp_get_num_threads();
    const int64_t K0 = 8 + ncol * t / nt;
    const int64_t W = ncol * (t + 1) / nt - (K0 - 8);
    if (W > 0) span_work(aa, bb, cc, N, K0, W);
  }
}
