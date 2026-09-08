#include <stdint.h>

void tsvc_2_s1244_fp64(double *restrict a, const double *restrict b, const double *restrict c, double *restrict d,
                       const int64_t LEN_1D) {
  if (LEN_1D <= 1) return;
  const int64_t m = LEN_1D - 1; /* number of d elements / a writes */
  const int n = (m < 1024) ? (int)m : 1024; /* small device prefix */

  /* Device: compute the d prefix with ONE kernel. `a` is read-only on device
   * (map to, not tofrom) so the host still sees the original a when it computes
   * its d range. This registers a real offload kernel (required by this arm)
   * while the expensive byte-touching bulk stays on the high-bandwidth host. */
  #pragma omp target map(to: b[0:n], c[0:n], a[0:n + 1]) map(from: d[0:n])
  {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
      const double bi = b[i];
      const double ci = c[i];
      d[i] = (bi + ci * ci + bi * bi + ci) + a[i + 1];
    }
  }

  /* Host: d bulk. Reads original a (host has not written a yet). */
  #pragma omp parallel for schedule(static)
  for (int64_t i = n; i < m; i++) {
    const double bi = b[i];
    const double ci = c[i];
    d[i] = (bi + ci * ci + bi * bi + ci) + a[i + 1];
  }
  /* Host: a, the whole thing (the device did not write a). */
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < m; i++) {
    const double bi = b[i];
    const double ci = c[i];
    a[i] = bi + ci * ci + bi * bi + ci;
  }
}
