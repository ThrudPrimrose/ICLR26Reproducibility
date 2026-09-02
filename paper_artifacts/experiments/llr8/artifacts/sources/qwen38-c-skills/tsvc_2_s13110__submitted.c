#include <stdint.h>
#include <limits.h>
#include <math.h>
#include <omp.h>

/* TSVC tsvc_2 s13110 (fp64 single-invocation): scan the LEN_2D x LEN_2D
 * matrix for the maximum, break ties at the FIRST position in row-major
 * order (reference uses strict '>' over the flattened array), and write
 * bb[0] = maxv + xindex + yindex.
 *
 * One flattened, fully parallel + SIMD pass over the flattened row-major
 * array. The reduction combines (value, position) pairs: larger value
 * wins; on equal values the smaller position wins. That comparator is
 * associative/commutative with identity (-HUGE_VAL, INT64_MAX), and its
 * result is exactly the reference's: the smallest flat index attaining
 * the global maximum.
 */

typedef struct { double val; int64_t pos; } mp_t;

static inline mp_t mp_choose(const mp_t a, const mp_t b) {
  if (a.val > b.val) return a;
  if (b.val > a.val) return b;
  return (a.pos <= b.pos) ? a : b;
}

/* private copies are seeded from the entry value of the reduction variable,
 * so the caller sets `best` to the identity below. */
#pragma omp declare reduction (mpmax: mp_t : omp_out = mp_choose(omp_out, omp_in))

void tsvc_2_s13110_fp64(const double *restrict aa, double *restrict bb, const int64_t LEN_2D) {
  const int64_t N = LEN_2D * LEN_2D;
  mp_t best = (mp_t){-HUGE_VAL, INT64_MAX};

#pragma omp parallel for simd reduction(mpmax: best) schedule(static)
  for (int64_t k = 0; k < N; ++k) {
    mp_t t = {aa[k], k};
    best = mp_choose(best, t);
  }

  const int64_t xindex = best.pos / LEN_2D;
  const int64_t yindex = best.pos % LEN_2D;
  bb[0] = best.val + (double)xindex + (double)yindex;
}
