#include <hip/hip_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_HIP(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::printf("HIP error %s at line %d\n", hipGetErrorString(e_), __LINE__); std::abort(); } } while(0)

__global__ void tsvc_kernel(double * __restrict__ a, const double * __restrict__ aa, int N) {
  int tid = threadIdx.x;
  int B = blockDim.x;
  for (int j = 0; j < N; j++) {
    double aj = a[j];
    for (int i = j + 1 + tid; i < N; i += B) {
      a[i] -= aa[(long long)j * N + i] * aj;
    }
    __syncthreads();
  }
}

extern "C" void tsvc_2_s115_fp64(double *a, const double *aa, const int64_t LEN_2D) {
  int N = (int)LEN_2D;
  int B = 256;
  tsvc_kernel<<<1, B>>>(a, aa, N);
  CHECK_HIP(hipGetLastError());
  CHECK_HIP(hipDeviceSynchronize());
}
