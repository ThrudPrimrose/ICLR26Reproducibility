#include <stdint.h>
#include <omp.h>

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N <= 8) return;
  const int64_t ncol = N - 8;              /* columns 8..N-1 */
  const int64_t full = ncol / 8;           /* groups of 8 consecutive columns */
  const int64_t tail = ncol - full * 8;    /* leftover columns */
  const int64_t nfull = full;

  #pragma omp parallel
  {
    #pragma omp for schedule(static)
    for (int64_t k = 0; k < nfull; ++k) {
      const int64_t c = 8 + 8 * k;         /* first column of this group */
      double sa0 = aa[7 * N + c + 0], sa1 = aa[7 * N + c + 1], sa2 = aa[7 * N + c + 2], sa3 = aa[7 * N + c + 3];
      double sa4 = aa[7 * N + c + 4], sa5 = aa[7 * N + c + 5], sa6 = aa[7 * N + c + 6], sa7 = aa[7 * N + c + 7];
      double sb0 = bb[7 * N + c + 0], sb1 = bb[7 * N + c + 1], sb2 = bb[7 * N + c + 2], sb3 = bb[7 * N + c + 3];
      double sb4 = bb[7 * N + c + 4], sb5 = bb[7 * N + c + 5], sb6 = bb[7 * N + c + 6], sb7 = bb[7 * N + c + 7];
      for (int64_t r = 8; r < N; ++r) {
        const double *cv = cc + r * N + c;
        double *av = aa + r * N + c;
        double *bv = bb + r * N + c;
        sa0 += cv[0]; sa1 += cv[1]; sa2 += cv[2]; sa3 += cv[3];
        sa4 += cv[4]; sa5 += cv[5]; sa6 += cv[6]; sa7 += cv[7];
        av[0] = sa0; av[1] = sa1; av[2] = sa2; av[3] = sa3;
        av[4] = sa4; av[5] = sa5; av[6] = sa6; av[7] = sa7;
        sb0 += cv[0]; sb1 += cv[1]; sb2 += cv[2]; sb3 += cv[3];
        sb4 += cv[4]; sb5 += cv[5]; sb6 += cv[6]; sb7 += cv[7];
        bv[0] = sb0; bv[1] = sb1; bv[2] = sb2; bv[3] = sb3;
        bv[4] = sb4; bv[5] = sb5; bv[6] = sb6; bv[7] = sb7;
      }
    }
    /* scalar tail columns */
    #pragma omp for schedule(static)
    for (int64_t t = 0; t < tail; ++t) {
      const int64_t c = 8 + 8 * full + t;
      double sa = aa[7 * N + c];
      double sb = bb[7 * N + c];
      for (int64_t r = 8; r < N; ++r) {
        const double v = cc[r * N + c];
        sa += v; aa[r * N + c] = sa;
        sb += v; bb[r * N + c] = sb;
      }
    }
  }
}
