#include <stdint.h>
#include <omp.h>

/* tsvc_2 s115: a[i] -= aa[j*N+i]*a[j] for j<i.
 *
 * This is a unit-triangular solve: a[j] is final exactly when step j reads
 * it, so the j loop carries a length-N chain and cannot be parallelized as a
 * whole.  We therefore run the chain one element per row (a[j+1] -= M[j,j+1]*a[j])
 * and, per block of B rows, fold the remaining B row-updates into a single
 * fused vector pass over a[i] (i >= j0+B) so each a[i] is RMW'd once per
 * block instead of once per row.  The fused pass is thread-parallel over i.
 */
void tsvc_2_s115_fp64(double *restrict a, const double *restrict aa, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N < 2) return;

  const int64_t B = 64;
  const int64_t NB = (N - 1 + B - 1) / B;

  #pragma omp parallel
  {
    for (int64_t p = 0; p < NB; p++) {
      const int64_t j0 = p * B;
      int64_t Bp = N - 1 - j0;
      if (Bp > B) Bp = B;
      const double *restrict base = aa + j0 * N;

      /* Subtriangle: finalize a[j0+1 .. j0+Bp-1].  Row-major, so the only
       * serial work is the diagonal step a[j0+r+1] (the chain itself); the
       * rest of row r is an independent (short) vector tail. */
      #pragma omp master
      {
        for (int64_t r = 0; r < Bp - 1; r++) {
          const double aj = a[j0 + r];
          const double *restrict row = base + r * N + j0 + r + 1;
          a[j0 + r + 1] -= row[0] * aj;
          for (int64_t k = 1; k + r < Bp - 1; k++)
            a[j0 + r + 1 + k] -= row[k] * aj;
        }
      }

      /* Bulk: a[i] -= sum_{r=0}^{Bp-1} M[j0+r][i] * a[j0+r], parallel over i. */
      #pragma omp for schedule(static)
      for (int64_t i = j0 + Bp; i < N; i++) {
        double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
        const double *restrict p0 = base + i;
        int64_t r = 0;
        for (; r + 3 < Bp; r += 4) {
          acc0 += p0[r * N]     * a[j0 + r];
          acc1 += p0[(r + 1) * N] * a[j0 + r + 1];
          acc2 += p0[(r + 2) * N] * a[j0 + r + 2];
          acc3 += p0[(r + 3) * N] * a[j0 + r + 3];
        }
        for (; r < Bp; r++)
          acc0 += p0[r * N] * a[j0 + r];
        a[i] -= (acc0 + acc1) + (acc2 + acc3);
      }
    }
  }
}
