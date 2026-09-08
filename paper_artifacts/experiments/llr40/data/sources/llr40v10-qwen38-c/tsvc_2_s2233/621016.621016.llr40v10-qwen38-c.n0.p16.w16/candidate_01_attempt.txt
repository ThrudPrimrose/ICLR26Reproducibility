#include <stdint.h>

/* tsvc_2 s2233: per-column dependent scans, independent across columns.
 * aa:  for each column i: acc = aa[7,i]; for j=8..N-1: acc += cc[j,i]; aa[j,i]=acc
 * bb:  for each column j: acc = bb[7,j]; for i=8..N-1: acc += cc[i,j]; bb[i,j]=acc
 * Same operations, same order => bit-identical to the reference.
 * Process 8 columns at once (contiguous rows) so the row loop vectorizes
 * (noinline helpers keep the vectorizer outside the OpenMP region, where
 * GCC's clone analysis misses the contiguous access); parallelize blocks. */

static __attribute__((noinline)) void aa_block(double *restrict aa, const double *restrict cc, int64_t N, int64_t i0) {
  double acc[8];
  const double *a0 = aa + 7 * N + i0;
  #pragma GCC unroll 8
  for (int64_t k = 0; k < 8; ++k) acc[k] = a0[k];
  for (int64_t j = 8; j < N; ++j) {
    const double *cref = cc + j * N + i0;
    double *aref = aa + j * N + i0;
    #pragma GCC unroll 8
    for (int64_t k = 0; k < 8; ++k) {
      acc[k] += cref[k];
      aref[k] = acc[k];
    }
  }
}

static __attribute__((noinline)) void bb_block(double *restrict bb, const double *restrict cc, int64_t N, int64_t j0) {
  double acc[8];
  const double *b0 = bb + 7 * N + j0;
  #pragma GCC unroll 8
  for (int64_t k = 0; k < 8; ++k) acc[k] = b0[k];
  for (int64_t i = 8; i < N; ++i) {
    const double *cref = cc + i * N + j0;
    double *bref = bb + i * N + j0;
    #pragma GCC unroll 8
    for (int64_t k = 0; k < 8; ++k) {
      acc[k] += cref[k];
      bref[k] = acc[k];
    }
  }
}

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N <= 8) return;
  const int64_t nfull = (N - 8) / 8;
  const int64_t tail = (N - 8) - 8 * nfull;
  const int64_t nb = nfull + (tail ? 1 : 0);

  #pragma omp parallel
  {
    #pragma omp for schedule(static)
    for (int64_t b = 0; b < nb; ++b) {
      if (b < nfull) aa_block(aa, cc, N, 8 + 8 * b);
      else {
        for (int64_t k = 0; k < tail; ++k) {
          const int64_t i = 8 + 8 * nfull + k;
          double r = aa[7 * N + i];
          for (int64_t j = 8; j < N; ++j) { r += cc[j * N + i]; aa[j * N + i] = r; }
        }
      }
    }
    #pragma omp for schedule(static)
    for (int64_t b = 0; b < nb; ++b) {
      if (b < nfull) bb_block(bb, cc, N, 8 + 8 * b);
      else {
        for (int64_t k = 0; k < tail; ++k) {
          const int64_t j = 8 + 8 * nfull + k;
          double r = bb[7 * N + j];
          for (int64_t i = 8; i < N; ++i) { r += cc[i * N + j]; bb[i * N + j] = r; }
        }
      }
    }
  }
}
