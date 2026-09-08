/* TSVC s255: a[i] = 0.333*(b[i]+b[i-1]+b[i-2]) with wraparound b[-1]=b[n-1], b[-2]=b[n-2].
 * The carry-around scalars x,y merely hold b[i-1],b[i-2], so the loop is dependency-free.
 * Small sizes run vectorized on the host; large sizes are offloaded to the GPU. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static void host3(const double *restrict b, double *restrict a, const int64_t n) {
  const double C = 0.333;
  if (n == 1) { a[0] = (b[0] + b[0] + b[0]) * C; return; }
  a[0] = (b[0] + b[n-1] + b[n-2]) * C;
  a[1] = (b[1] + b[0] + b[n-1]) * C;
  for (int64_t i = 2; i < n; i++) {
    a[i] = (b[i] + b[i-1] + b[i-2]) * C;
  }
}

void tsvc_2_s255_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
  static int probed = 0;
  if (!probed && LEN_1D > (1 << 20)) {
    probed = 1;
    fprintf(stdout, "PROBE LEN=%lld ndev=%d maxthr=%d "
      "HSA_VISIBLE=%s HIP_VISIBLE=%s GPU_ORD=%s XNACK=%s OFFLOAD=%s\n",
      (long long)LEN_1D, omp_get_num_devices(), omp_get_max_threads(),
      getenv("HSA_VISIBLE_DEVICES") ? getenv("HSA_VISIBLE_DEVICES") : "(null)",
      getenv("HIP_VISIBLE_DEVICES") ? getenv("HIP_VISIBLE_DEVICES") : "(null)",
      getenv("GPU_DEVICE_ORDINAL") ? getenv("GPU_DEVICE_ORDINAL") : "(null)",
      getenv("HSA_XNACK") ? getenv("HSA_XNACK") : "(null)",
      getenv("OMP_TARGET_OFFLOAD") ? getenv("OMP_TARGET_OFFLOAD") : "(null)");
    fflush(stdout);
  }
  if (LEN_1D <= 0) return;
  const double C = 0.333;
  if (LEN_1D <= (1 << 20)) {
    host3(b, a, LEN_1D);
    return;
  }
  #pragma omp target map(to: b[0:LEN_1D]) map(from: a[0:LEN_1D])
  {
    #pragma omp teams distribute
    for (int64_t i = 0; i < LEN_1D; i++) {
      if (i == 0) a[i] = (b[i] + b[LEN_1D-1] + b[LEN_1D-2]) * C;
      else if (i == 1) a[i] = (b[i] + b[i-1] + b[LEN_1D-1]) * C;
      else a[i] = (b[i] + b[i-1] + b[i-2]) * C;
    }
  }
}
