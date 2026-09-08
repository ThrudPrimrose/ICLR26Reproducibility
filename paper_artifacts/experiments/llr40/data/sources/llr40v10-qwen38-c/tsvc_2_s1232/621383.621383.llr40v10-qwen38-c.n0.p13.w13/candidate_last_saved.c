#include <stdint.h>
#include <omp.h>

/* TSVC tsvc_2 s1232: aa[i,j] = bb[i,j] + cc[i,j] for j in [0,N), i in [j*VLEN, N).
 * Row i owns the contiguous columns j in [0, L_i) where
 *   L_i = min(cap, i/V + 1)  (V > 0, cap = ceil(N/V))   -- dense rows if V <= 0.
 * Rows are independent; each row's slice is contiguous, so the inner loop vectorizes.
 * Threads own ranges of rows chosen so each thread does ~W/T elements (balanced). */

static int64_t row_len(int64_t i, int64_t N, int64_t V) { /* V > 0 */
  int64_t cap = (N + V - 1) / V;
  int64_t nj = i / V + 1;
  return nj < cap ? nj : cap;
}

/* F(i) = sum_{i' < i} L_{i'} for V > 0;  F(0) = 0,  F(N) = total work W. */
static int64_t prefix_of(int64_t i, int64_t N, int64_t V) {
  int64_t C = (N + V - 1) / V;          /* cap */
  int64_t stop = (C - 1) * V;           /* first row with the capped length */
  if (i > stop)
    return V * (C - 1) * (C - 2) / 2 + (C - 1) * V + (i - stop) * C;
  int64_t k = i / V, r = i % V;
  return V * k * (k - 1) / 2 + r * k + i;
}

/* smallest i in [0, N] with prefix_of(i) > t   (exists since prefix_of(N) = W > t) */
static int64_t find_cut(int64_t t, int64_t N, int64_t V) {
  int64_t lo = 0, hi = N;
  while (hi - lo > 1) {
    int64_t mid = lo + (hi - lo) / 2;
    if (prefix_of(mid, N, V) > t) hi = mid;
    else lo = mid;
  }
  return lo;
}

void tsvc_2_s1232_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc,
                       const int64_t LEN_2D, const int64_t VLEN) {
  const int64_t N = LEN_2D, V = VLEN;
  if (N <= 0) return;
  #pragma omp parallel
  {
    const int nt = omp_get_num_threads();
    const int tid = omp_get_thread_num();
    int64_t i0, i1;
    if (V <= 0) {
      i0 = N * (int64_t)tid / nt;
      i1 = N * (int64_t)(tid + 1) / nt;
    } else {
      const int64_t W = prefix_of(N, N, V);
      i0 = (tid == 0) ? 0 : find_cut(W * (int64_t)tid / nt, N, V);
      i1 = (tid == nt - 1) ? N : find_cut(W * (int64_t)(tid + 1) / nt, N, V);
    }
    for (int64_t i = i0; i < i1; ++i) {
      const int64_t nj = (V <= 0) ? N : row_len(i, N, V);
      double *ra = aa + i * N;
      const double *rb = bb + i * N;
      const double *rc = cc + i * N;
      for (int64_t j = 0; j < nj; ++j) {
        ra[j] = rb[j] + rc[j];
      }
    }
  }
}
