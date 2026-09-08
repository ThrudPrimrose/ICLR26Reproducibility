#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(expr)                                                    \
  do {                                                                     \
    hipError_t _e = (expr);                                                \
    if (_e != hipSuccess) {                                                \
      fprintf(stderr, "HIP error %s (%s:%d)\n", hipGetErrorString(_e),     \
              __FILE__, __LINE__);                                         \
      abort();                                                             \
    }                                                                      \
  } while (0)

namespace {

template <int DO_B>
__global__ void __launch_bounds__(256) fmif_vec(double4 *a4, double4 *b4,
                                                const double *cond, const double4 *src4,
                                                int64_t LEN_2D) {
  const int64_t row = blockIdx.y;
  const int row4 = (int)(LEN_2D >> 2);
  const bool doA = cond[row] > 0.0;
  double4 *ar = a4 + row * row4;
  const double4 *sr = src4 + row * row4;
  double4 *br = b4 + row * row4;
  for (int j = threadIdx.x; j < row4; j += blockDim.x) {
    const double4 s = sr[j];
    if (doA) {
      double4 t;
      t.x = s.x * 2.0;
      t.y = s.y * 2.0;
      t.z = s.z * 2.0;
      t.w = s.w * 2.0;
      ar[j] = t;
    }
    if (DO_B) {
      double4 t;
      t.x = s.x + 1.0;
      t.y = s.y + 1.0;
      t.z = s.z + 1.0;
      t.w = s.w + 1.0;
      br[j] = t;
    }
  }
}

template <int DO_B>
__global__ void __launch_bounds__(256) fmif_scalar(double *a, double *b, const double *cond,
                                                   const double *src, int64_t LEN_2D) {
  const int64_t row = blockIdx.y;
  const bool doA = cond[row] > 0.0;
  double *ar = a + row * LEN_2D;
  const double *sr = src + row * LEN_2D;
  double *br = b + row * LEN_2D;
  for (int64_t j = threadIdx.x; j < LEN_2D; j += blockDim.x) {
    const double s = sr[j];
    if (doA) ar[j] = s * 2.0;
    if (DO_B) br[j] = s + 1.0;
  }
}

template <int DO_B>
__global__ void fmif_flat(double *a, double *b, const double *cond, const double *src,
                          int64_t LEN_2D) {
  const int64_t total = LEN_2D * LEN_2D;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
    const double s = src[idx];
    if (cond[idx / LEN_2D] > 0.0) a[idx] = s * 2.0;
    if (DO_B) b[idx] = s + 1.0;
  }
}

} // namespace

void fmif_launch(double *a, double *b, const double *cond, const double *src,
                 const int64_t K, const int64_t LEN_2D) {
  if (LEN_2D <= 0) return;
  const int kt = 256;
  if (LEN_2D > 65535) {
    const int64_t total = LEN_2D * LEN_2D;
    int64_t blocks = (total + kt - 1) / kt;
    if (blocks > 16384) blocks = 16384;
    if (blocks < 1) blocks = 1;
    dim3 g((unsigned)blocks);
    if (K > 0) fmif_flat<1><<<g, kt>>>(a, b, cond, src, LEN_2D);
    else fmif_flat<0><<<g, kt>>>(a, b, cond, src, LEN_2D);
  } else if ((LEN_2D & 3) == 0) {
    dim3 g(1, (unsigned)LEN_2D);
    if (K > 0)
      fmif_vec<1><<<g, kt>>>((double4 *)a, (double4 *)b, cond, (const double4 *)src, LEN_2D);
    else
      fmif_vec<0><<<g, kt>>>((double4 *)a, (double4 *)b, cond, (const double4 *)src, LEN_2D);
  } else {
    dim3 g(1, (unsigned)LEN_2D);
    if (K > 0) fmif_scalar<1><<<g, kt>>>(a, b, cond, src, LEN_2D);
    else fmif_scalar<0><<<g, kt>>>(a, b, cond, src, LEN_2D);
  }
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
}
