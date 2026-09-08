#include <cstdint>
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>

extern "C" void tsvc_2_vpvts_fp64(double* __restrict__ a, const double* __restrict__ b, const int64_t LEN_1D, const int64_t S) {
    double s = static_cast<double>(S);
    const int blockSize = 256;
    const int gridSize = static_cast<int>((LEN_1D + blockSize - 1) / blockSize);
    hipLaunchKernelGGL(tsvc_2_vpvts_kernel, dim3(gridSize), dim3(blockSize), 0, 0, a, b, LEN_1D, s);
}

__global__ void tsvc_2_vpvts_kernel(double* __restrict__ a, const double* __restrict__ b, const int64_t LEN_1D, const double s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < LEN_1D) {
        a[idx] += b[idx] * s;
    }
}
#else
extern "C" void tsvc_2_vpvts_fp64(double* a, const double* b, const int64_t LEN_1D, const int64_t S) {
    double s = static_cast<double>(S);
    for (int64_t i = 0; i < LEN_1D; ++i) {
        a[i] += b[i] * s;
    }
}
#endif
