// Device half: kernels + launcher. Built by the judge as a separate translation unit.
#include <hip/hip_runtime.h>
#include <cstdint>

#define TSS_BLOCK 256

__global__ void __launch_bounds__(TSS_BLOCK)
tss_k1(const double* __restrict__ a, double* __restrict__ partials,
       const int64_t n, const int ws) {
  double s = 0.0;
  const int64_t stride = (int64_t)blockDim.x * gridDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const double v = a[i];
    s += (v > 0.0) ? v : 0.0;
  }
  // deterministic block reduction
  __shared__ double sh[TSS_BLOCK / 32];
  const int lane = threadIdx.x % ws;
  const int wid = threadIdx.x / ws;
  for (int o = ws >> 1; o > 0; o >>= 1) s += __shfl_down(s, o, ws);
  if (lane == 0) sh[wid] = s;
  __syncthreads();
  const int nw = TSS_BLOCK / ws;
  if (wid == 0) {
    s = (lane < nw) ? sh[lane] : 0.0;
    for (int o = 1; o < nw; o <<= 1) s += __shfl_down(s, o, ws);
    if (lane == 0) partials[blockIdx.x] = s;
  }
}

__global__ void __launch_bounds__(TSS_BLOCK)
tss_k2(const double* __restrict__ partials, double* __restrict__ b,
       const int64_t np, const int ws) {
  double s = 0.0;
  for (int64_t i = threadIdx.x; i < np; i += blockDim.x) s += partials[i];
  __shared__ double sh[TSS_BLOCK / 32];
  const int lane = threadIdx.x % ws;
  const int wid = threadIdx.x / ws;
  for (int o = ws >> 1; o > 0; o >>= 1) s += __shfl_down(s, o, ws);
  if (lane == 0) sh[wid] = s;
  __syncthreads();
  const int nw = TSS_BLOCK / ws;
  if (wid == 0) {
    s = (lane < nw) ? sh[lane] : 0.0;
    for (int o = 1; o < nw; o <<= 1) s += __shfl_down(s, o, ws);
    if (lane == 0) b[0] = s;
  }
}

extern "C" void tss2_launch(const double* a, double* b, const int64_t n, hipStream_t stream) {
  static double* g_partials = nullptr;
  static int64_t g_part_cap = 0;
  static int g_ws = 0, g_nsm = 0;
  if (!g_partials) {
    int dev = 0;
    hipError_t e = hipGetDevice(&dev);
    if (e != hipSuccess) { std::fprintf(stderr, "hipGetDevice %s\n", hipGetErrorString(e)); std::abort(); }
    e = hipDeviceGetAttribute(&g_nsm, hipDeviceAttributeMultiprocessorCount, dev);
    if (e != hipSuccess) { std::fprintf(stderr, "attr nsm %s\n", hipGetErrorString(e)); std::abort(); }
    e = hipDeviceGetAttribute(&g_ws, hipDeviceAttributeWarpSize, dev);
    if (e != hipSuccess) { std::fprintf(stderr, "attr ws %s\n", hipGetErrorString(e)); std::abort(); }
    g_part_cap = 16384;
    e = hipMalloc(&g_partials, (size_t)g_part_cap * sizeof(double));
    if (e != hipSuccess) { std::fprintf(stderr, "hipMalloc %s\n", hipGetErrorString(e)); std::abort(); }
  }
  int64_t grid = (n + TSS_BLOCK - 1) / TSS_BLOCK;
  const int64_t cap = (int64_t)g_nsm * 8;
  if (grid > cap) grid = cap;
  if (grid < 1) grid = 1;
  if (grid > g_part_cap) grid = g_part_cap;
  tss_k1<<<(unsigned)grid, TSS_BLOCK, 0, stream>>>(a, g_partials, n, g_ws);
  tss_k2<<<1, TSS_BLOCK, 0, stream>>>(g_partials, b, grid, g_ws);
}
// Host half: the C-ABI entry point. Does nothing but launch.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" void tss2_launch(const double* a, double* b, const int64_t n, hipStream_t stream);

extern "C" void tsvc_2_s3111_fp64(const double* __restrict__ a, double* __restrict__ b,
                                  const int64_t LEN_1D) {
  tss2_launch(a, b, LEN_1D, (hipStream_t)0);
  hipError_t e = hipGetLastError();
  if (e != hipSuccess) { std::fprintf(stderr, "launch error %s\n", hipGetErrorString(e)); std::abort(); }
}
