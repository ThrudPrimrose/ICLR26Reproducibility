#include <stdint.h>
#include <omp.h>

/*
 * TSVC s115: for j: for i>j: a[i] -= aa[j,i] * a[j].
 *
 * The j loop is a true dependence chain (a[j] is read after all earlier rows
 * updated it); inside one row every i update is independent.  We keep the
 * reference per-element j-order exactly (each element accumulates its
 * contributions in increasing j order, one FMA per contribution), so results
 * are bit-identical to the serial reference.
 *
 * Balanced recursion over the index range [lo, hi):
 *   1. left half solved fully (recursively),
 *   2. cross block:  a[i] -= sum_{j in left} aa[j,i]*a[j]  for i in the right
 *      half -- rows j are independent now, parallel over i;
 *   3. right half solved (recursively).
 * Every aa[j,i] is applied exactly once: total work n(n-1)/2, depth O(log n).
 *
 * All threads walk the same recursion tree in lockstep; work is split at the
 * cross matvec (an `omp for`) and each leaf body is executed by one
 * round-robin thread (its rows are a serial chain) followed by a barrier.
 */

#define LEAF 512
#define PF_DIST 16

static void solve_leaf(double *a, const double *aa, int64_t n,
                       int64_t lo, int64_t hi) {
  for (int64_t j = lo; j < hi; j++) {
    const double aj = a[j];
    const double *row = aa + j * n;
    for (int64_t i = j + 1; i < hi; i++)
      a[i] -= row[i] * aj;
  }
}

static void solve_leaf_shared(double *a, const double *aa, int64_t n,
                              int64_t lo, int64_t hi) {
  const int tid = omp_get_thread_num();
  const int nt = omp_get_num_threads();
  const unsigned h = (unsigned)lo * 2654435761u ^ (unsigned)hi * 40503u;
  if ((h & 1023u) % (unsigned)nt == (unsigned)tid)
    solve_leaf(a, aa, n, lo, hi);
  #pragma omp barrier
}

static void solve_range(double *a, const double *aa, int64_t n,
                        int64_t lo, int64_t hi) {
  const int64_t len = hi - lo;
  if (len <= LEAF) {
    solve_leaf_shared(a, aa, n, lo, hi);
    return;
  }
  const int64_t mid = (lo + hi) >> 1;
  solve_range(a, aa, n, lo, mid);

  /* cross matvec, parallel over i, 8-wide groups; bit-exact per element:
     each j applies one FMA per element, in increasing j order */
  const int64_t i_end = hi & ~(int64_t)7;
#pragma omp for schedule(static)
  for (int64_t i = mid; i < i_end; i += 8) {
    for (int64_t j = lo; j < mid; j++) {
      if ((j & (PF_DIST - 1)) == 0)
        __builtin_prefetch(aa + (j + PF_DIST) * n + i, 0, 1);
      const double *p = aa + j * n + i;
      const double xj = a[j];
      a[i + 0] -= p[0] * xj;
      a[i + 1] -= p[1] * xj;
      a[i + 2] -= p[2] * xj;
      a[i + 3] -= p[3] * xj;
      a[i + 4] -= p[4] * xj;
      a[i + 5] -= p[5] * xj;
      a[i + 6] -= p[6] * xj;
      a[i + 7] -= p[7] * xj;
    }
  }
  /* tail */
  const int64_t t0 = i_end > mid ? i_end : mid;
#pragma omp for schedule(static)
  for (int64_t i = t0; i < hi; i++) {
    for (int64_t j = lo; j < mid; j++)
      a[i] -= aa[j * n + i] * a[j];
  }

  solve_range(a, aa, n, mid, hi);
}

void tsvc_2_s115_fp64(double *restrict a, const double *restrict aa,
                      const int64_t LEN_2D) {
  const int64_t n = LEN_2D;
  if (n <= 4096) {
    solve_leaf(a, aa, n, 0, n);
    return;
  }
#pragma omp parallel
  {
    solve_range(a, aa, n, 0, n);
  }
}
