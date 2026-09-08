#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

// Simple error handling macro
#define HIP_CHECK(err) \
    do { \
        hipError_t e = (err); \
        if (e != hipSuccess) { \
            fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
            std::abort(); \
        } \
    } while (0)

// Device kernel performing a[i] += b[i] * S
__global__ void tsvc_2_vpvts_kernel(double* __restrict__ a,
                                   const double* __restrict__ b,
                                   int64_t LEN_1D,
                                   int64_t S) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < LEN_1D; i += stride) {
        a[i] += b[i] * static_cast<double>(S);
    }
}

// Host entry point matching the C-ABI expected by the benchmark.
extern "C" void tsvc_2_vpvts_fp64(double* a, const double* b, int64_t LEN_1D, int64_t S) {
    // Choose launch configuration
    const int threadsPerBlock = 256;
    int blocks = static_cast<int>((LEN_1D + threadsPerBlock - 1) / threadsPerBlock);
    if (blocks < 1) blocks = 1;
    // Launch kernel
    hipLaunchKernelGGL(tsvc_2_vpvts_kernel,
                       dim3(blocks), dim3(threadsPerBlock), 0, 0,
                       a, b, LEN_1D, S);
    // Check for launch errors
    HIP_CHECK(hipGetLastError());
    // Ensure completion
    HIP_CHECK(hipDeviceSynchronize());
}

