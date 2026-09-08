/* scatter_accum_dup: bins[ip[i]] += src[i]; ip may repeat indices.
 * ABI (judge): (double *bins, const int32_t *ip, const double *src, int LEN_1D)
 * v3: permutation fast path + parallel atomic path. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define NT 120

void scatter_accum_dup_fp64(double *bins, const int32_t *ip, const double *src, int LEN_1D) {
  int N = LEN_1D;
  if (N <= 0) return;
  int nt = omp_get_max_threads();
  if (nt > NT) nt = NT;
  if (N < 100000) nt = (N < 10000) ? 1 : 8;
  int use_threads = (N >= 4000000) ? nt : (N >= 100000 ? nt / 2 : 1);
  if (use_threads < 1) use_threads = 1;

  int is_perm = 0;
  if (N >= 100000) {
    is_perm = 1;
    size_t nwords = ((size_t)N + 7) >> 3;
    uint8_t *seen = malloc(nwords);
    if (seen) {
      memset(seen, 0, nwords);
#pragma omp parallel for num_threads(use_threads) schedule(static, 2048) reduction(&:is_perm)
      for (int i = 0; i < N; ++i) {
        int j = ip[i];
        uint8_t *s = seen + ((size_t)j >> 3);
        uint8_t m = (uint8_t)(1u << (j & 7));
        if (*s & m) is_perm = 0;
        *s = (uint8_t)(*s | m);
      }
      free(seen);
    }
  }

  if (is_perm) {
#pragma omp parallel for num_threads(use_threads) schedule(static, 2048)
    for (int i = 0; i < N; ++i) bins[ip[i]] += src[i];
  } else {
#pragma omp parallel for num_threads(use_threads) schedule(static, 2048)
    for (int i = 0; i < N; ++i) {
      int j = ip[i];
#pragma omp atomic update
      bins[j] += src[i];
    }
  }
}
