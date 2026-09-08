#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

template <typename T>
__global__ void tsvc_2_s4112_kernel(T* __restrict__ a, const T* __restrict__ b, const int32_t* __restrict__ ip, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        a[i] += b[ip[i]] * static_cast<T>(2.0);
    }
}

extern "C" void tsvc_2_s4112_fp64(double* __restrict__ a, const double* __restrict__ b,
                                   const int32_t* __restrict__ ip, int64_t LEN_1D) {
    const int blockSize = 256;
    int gridSize = static_cast<int>((LEN_1D + blockSize - 1) / blockSize);
    if (gridSize > 65535) gridSize = 65535;
    hipLaunchKernelGGL(tsvc_2_s4112_kernel<double>, dim3(gridSize), dim3(blockSize), 0, 0,
                       a, b, ip, LEN_1D);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        const char* msg = hipGetErrorString(err);
        fprintf(stderr, "HIP kernel launch failed: %s\n", msg);
        abort();
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        const char* msg = hipGetErrorString(err);
        fprintf(stderr, "HIP device sync failed: %s\n", msg);
        abort();
    }
}
