#include <stdint.h>

/* TSVC tsvc_2 s1232: aa[i,j] = bb[i,j] + cc[i,j] for i in [j*VLEN, LEN_2D), j in [0, LEN_2D).
 *
 * Reference nest iterates j outer / i inner, which is a STRIDED (LEN_2D) access in the
 * (i*N + j) layout and never vectorizes.  The region is a per-row prefix:
 *   (i,j) in region  <=>  j*VLEN <= i  <=>  j <= floor(i / VLEN)   (VLEN > 0)
 * so for fixed i the row is written unit-stride over j in [0, min(N-1, i/VLEN)].
 *
 * Dependences: every element (i,j) is written exactly once and read exactly once per
 * input array -- no dependence vector exists at all, so the nest is fully parallel and
 * the i/j interchange is trivially legal.  Rows with i >= (N-1)*VLEN are full rows.
 *
 * Rows carry growing work (jmax ~ i/VLEN), so chunk-1 static distribution keeps the
 * load balanced; the inner j loop is unit-stride over restrict pointers and
 * vectorizes; the outer i loop is threaded across cores. */

void tsvc_2_s1232_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D,
                       const int64_t VLEN) {
  const int64_t N = LEN_2D;
  if (N <= 0) return;

  /* first row index whose j-range is complete (jmax == N-1); N means none */
  int64_t i_full;
  if (VLEN <= 0) {
    i_full = 0; /* every row full */
  } else if (VLEN >= N) {
    i_full = N; /* i/VLEN < 1 for all i < N, so no full row */
  } else {
    i_full = (N - 1) * VLEN;
    if (i_full > N) i_full = N;
  }

  #pragma omp parallel
  {
    #pragma omp for schedule(static, 1)
    for (int64_t i = 0; i < i_full; ++i) {
      const int64_t jmax = i / VLEN; /* <= N-2 here */
      const double *restrict bbp = bb + i * N;
      const double *restrict ccp = cc + i * N;
      double *restrict aap = aa + i * N;
      for (int64_t j = 0; j <= jmax; ++j) {
        aap[j] = bbp[j] + ccp[j];
      }
    }
    #pragma omp for schedule(static, 1)
    for (int64_t i = i_full; i < N; ++i) {
      const double *restrict bbp = bb + i * N;
      const double *restrict ccp = cc + i * N;
      double *restrict aap = aa + i * N;
      for (int64_t j = 0; j < N; ++j) {
        aap[j] = bbp[j] + ccp[j];
      }
    }
  }
}
