#include <stdint.h>
#include <stdio.h>

/* Hand port of the TSVC tsvc_2 C++ microkernel ``s1232`` (s1232_d_single.cpp), fp64
 * single-invocation variant, to C23 under the v2 C-ABI. */
void tsvc_2_s1232_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D,
                       const int64_t VLEN) {

  for (int64_t j = 0; j < LEN_2D; ++j) {
    for (int64_t i = j * VLEN; i < LEN_2D; ++i) {
      aa[i * LEN_2D + j] = bb[i * LEN_2D + j] + cc[i * LEN_2D + j];
    }
  }
}
