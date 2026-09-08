#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Device kernel performing element-wise multiplication: a[i] = a[i] * b[i] * c[i]
__global__ void vtvtv_kernel(double* __restrict__ a, const double* __restrict__ b,
                              const double* __restrict__ c, int64_t LEN_1D) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < LEN_1D) {
        a[idx] = a[idx] * b[idx] * c[idx];
    }
}

extern "C" void tsvc_2_vtvtv_fp64(double* __restrict__ a, const double* __restrict__ b,
                                   const double* __restrict__ c, const int64_t LEN_1D) {
    // Choose a reasonable block size; 256 works well on most GPUs.
    const int threads_per_block = 256;
    int blocks = static_cast<int>((LEN_1D + threads_per_block - 1) / threads_per_block);
    if (blocks < 1) blocks = 1;

    // Launch the kernel.
    hipLaunchKernelGGL(vtvtv_kernel,
                       dim3(blocks), dim3(threads_per_block), 0, 0,
                       a, b, c, LEN_1D);
    // Check for launch errors.
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::fprintf(stderr, "HIP launch error: %s\n", hipGetErrorString(err));
        std::exit(1);
    }
    // Ensure the kernel has completed before returning.
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::fprintf(stderr, "HIP sync error: %s\n", hipGetErrorString(err));
        std::exit(1);
    }
}
