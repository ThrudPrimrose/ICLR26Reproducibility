#include <stdint.h>
#include <omp.h>

void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D){
  const int64_t L = LEN_2D;
  /* trivial device registration */
  int devflag = 0;
  #pragma omp target map(tofrom: devflag)
  devflag = 1;

  const int64_t GROUP = 64;
  #pragma omp parallel for schedule(static)
  for (int64_t g = 0; g < L; g += GROUP){
    int64_t g0 = g;
    int64_t gw = (g0 + GROUP <= L) ? GROUP : (L - g0);
    double acc[GROUP];
    for (int64_t c=0;c<gw;++c) acc[c] = aa[g0 + c];         /* row 0 */
    for (int64_t j=1;j<L;++j){
      const double *brow = bb + j*L + g0;
      double *arow  = aa + j*L + g0;
      for (int64_t c=0;c<gw;++c){ acc[c] += brow[c]; arow[c] = acc[c]; }
    }
  }
}
