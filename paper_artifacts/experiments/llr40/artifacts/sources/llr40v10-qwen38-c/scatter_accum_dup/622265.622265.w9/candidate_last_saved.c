/* scatter_accum_dup: bins[ip[i]] += src[i], ip may repeat indices.
 * ABI: (double *bins, const int32_t *ip, const double *src, int LEN_1D).
 * v2: runtime permutation detection; conflict-free path or parallel atomic path. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void scatter_accum_dup_fp64(double *bins, const int32_t *ip, const double *src, int LEN_1D) {
  if (LEN_1D <= 0) return;

  int is_perm = 1;
  size_t nwords = ((size_t)LEN_1D + 7) >> 3;
  uint8_t *seen = malloc(nwords ? nwords : 1);
  if (seen) {
    memset(seen, 0, nwords ? nwords : 1);
#pragma omp parallel for schedule(static, 2048) reduction(&:is_perm)
    for (int i = 0; i < LEN_1D; ++i) {
      int j = ip[i];
      uint8_t *s = seen + ((size_t)j >> 3);
      uint8_t m = (uint8_t)(1u << (j & 7));
      if (*s & m) is_perm = 0;
      *s = (uint8_t)(*s | m);
    }
    free(seen);
  }

  if (is_perm) {
#pragma omp parallel for schedule(static, 2048)
    for (int i = 0; i < LEN_1D; ++i) bins[ip[i]] += src[i];
  } else {
#pragma omp parallel for schedule(static, 2048)
    for (int i = 0; i < LEN_1D; ++i) {
      int j = ip[i];
#pragma omp atomic update
      bins[j] += src[i];
    }
  }
}
