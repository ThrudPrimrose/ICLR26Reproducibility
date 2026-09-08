/* scan_affine_decay: y[i] = c[i]*y[i-1] + x[i]  (y[0] = x[0] seed)
 *
 * Parallel blocked scan over the associative affine composition:
 * element i is the map M_i(t) = c[i]*t + x[i]; block maps compose as
 * (A,B).(a,b) = (A*a, A*b + B).  Phase 1 computes per-block (A_tot,B_tot);
 * phase 2 scans the (tiny) block totals; phase 3 reruns the recurrence per
 * block from the corrected entry value.  Traffic: 2 reads of c and x, 1
 * write of y = 40n bytes, no large temporaries.
 */
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

void scan_affine_decay_fp64(double *restrict c, double *restrict x,
                            double *restrict y, const int64_t LEN_1D,
                            uint8_t *workspace, int64_t workspace_size) {
  (void)workspace;
  (void)workspace_size;
  const int64_t n = LEN_1D;
  if (n <= 0) return;
  y[0] = x[0];
  if (n == 1) return;

  const int nt = omp_get_max_threads();
  int64_t nb = (int64_t)nt * 16;
  if (nb > n - 1) nb = n - 1;
  if (nb < 1) nb = 1;

  const int64_t span = n - 1;            /* elements 1..n-1 */
  const int64_t bsize = span / nb;
  const int64_t rem = span % nb;

  double *totA = malloc(nb * sizeof(double));
  double *totB = malloc(nb * sizeof(double));
  double *inA = malloc((nb + 1) * sizeof(double));
  double *inB = malloc((nb + 1) * sizeof(double));

  /* block b covers [s, e) of indices: first `rem` blocks have bsize+1 elems */

  /* phase 1: per-block totals (t: local B value, a: local A product) */
  #pragma omp parallel for schedule(static)
  for (int64_t b = 0; b < nb; b++) {
    int64_t s = 1 + b * bsize + (b < rem ? b : rem);
    int64_t e = s + bsize + (b < rem ? 1 : 0);
    double t = 0.0, a = 1.0;
    for (int64_t i = s; i < e; i++) {
      const double ci = c[i];
      t = ci * t + x[i];
      a *= ci;
    }
    totA[b] = a;
    totB[b] = t;
  }

  /* phase 2: entry value for each block: inA[b]*y0 + inB[b] = y[s_b - 1] */
  inA[0] = 1.0;
  inB[0] = 0.0;
  for (int64_t b = 1; b <= nb; b++) {
    inA[b] = totA[b - 1] * inA[b - 1];
    inB[b] = totA[b - 1] * inB[b - 1] + totB[b - 1];
  }
  const double y0 = y[0];

  /* phase 3: rerun the recurrence per block from the entry value */
  #pragma omp parallel for schedule(static)
  for (int64_t b = 0; b < nb; b++) {
    int64_t s = 1 + b * bsize + (b < rem ? b : rem);
    int64_t e = s + bsize + (b < rem ? 1 : 0);
    double t = inA[b] * y0 + inB[b];
    for (int64_t i = s; i < e; i++) {
      t = c[i] * t + x[i];
      y[i] = t;
    }
  }

  free(totA); free(totB); free(inA); free(inB);
}
