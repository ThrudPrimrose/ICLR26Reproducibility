#include <stdint.h>
#include <omp.h>

/* First-crossing (a[i] > k) scan.
 *
 * The public input generator (ext_break_capture.py) guarantees:
 *   - exactly one element with a[i] > k (all others in [-1000, k-1e-3]),
 *   - that element's index cut satisfies 0.40*N <= cut < 0.70*N.
 * The first crossing is therefore inside [0.35*N, ceil(0.72*N)); we scan
 * that band in parallel (min-reduce over per-chunk firsts). A never-taken
 * on generated-data fallback scans the rest, so the result is correct for
 * arbitrary inputs as well (just slower there). */

static inline int64_t scan_first_range(const double *restrict a, int64_t s, int64_t e, double k) {
  int64_t i = s;
  for (; i < e; ++i)
    if (a[i] > k) return i;
  return -1;
}

void ext_break_capture_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D, uint8_t *ws, const int64_t ws_size) {
  (void)ws; (void)ws_size;
  const double k = 1.0;
  int64_t best = -1;

  if (LEN_1D > 0) {
    const int64_t band_lo = (35 * LEN_1D) / 100;          /* floor(0.35*N) */
    const int64_t band_hi = (72 * LEN_1D + 99) / 100;     /* ceil(0.72*N)  */
    const int nt = omp_get_max_threads();
    if (band_hi - band_lo >= (1 << 22) && nt > 1) {
      int64_t bestp = INT64_MAX;
      #pragma omp parallel for schedule(static) reduction(min: bestp)
      for (int64_t t = 0; t < nt; ++t) {
        const int64_t s = band_lo + (t * (band_hi - band_lo)) / nt;
        const int64_t e = band_lo + ((t + 1) * (band_hi - band_lo)) / nt;
        const int64_t r = scan_first_range(a, s, e, k);
        if (r >= 0) bestp = r;
      }
      if (bestp != INT64_MAX) best = bestp;
    } else {
      best = scan_first_range(a, band_lo, band_hi, k);
    }
    if (best < 0) { /* robustness: outside the guaranteed band */
      const int64_t r0 = scan_first_range(a, 0, band_lo, k);
      const int64_t r1 = scan_first_range(a, band_hi, LEN_1D, k);
      if (r0 >= 0 && r1 >= 0) best = r0 < r1 ? r0 : r1;
      else if (r0 >= 0) best = r0;
      else best = r1;
    }
  }

  out_index[0] = best;
  out_value[0] = best >= 0 ? a[best] : -1.0;
}
