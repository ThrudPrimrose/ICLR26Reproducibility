// Host entry for tsvc_2_s1232 (HIP). C-ABI symbol the judge links against.
#include <stdint.h>

namespace s1232 {
void launch(double *aa, const double *bb, const double *cc, int64_t N, int64_t V);
}

extern "C" void tsvc_2_s1232_fp64(double *__restrict__ aa,
                                  const double *__restrict__ bb,
                                  const double *__restrict__ cc,
                                  const int64_t LEN_2D,
                                  const int64_t VLEN,
                                  void *__restrict__ workspace,
                                  const int64_t workspace_size) {
    (void)workspace;
    (void)workspace_size;
    s1232::launch(aa, bb, cc, LEN_2D, VLEN);
}
