#include <hip/hip_runtime.h>
#include <cstdint>

__global__ void tsvc_2_vag_kernel(double* __restrict__ a, const double* __restrict__ b, const int32_t* __restrict__ ip, int64_t LEN_1D);

extern "C" void tsvc_2_vag_fp64(double* __restrict__ a, const double* __restrict__ b, const int32_t* __restrict__ ip, const int64_t LEN_1D) {
    // Choose a reasonable block size. 256 threads per block is typical.
    const int blockSize = 256;
    int64_t gridSize = (LEN_1D + blockSize - 1) / blockSize;
    // Launch the kernel. Use hipLaunchKernelGGL which automatically calculates launch parameters.
    hipLaunchKernelGGL(tsvc_2_vag_kernel,
                       dim3(static_cast<unsigned int>(gridSize)), dim3(blockSize), 0, 0,
                       a, b, ip, LEN_1D);
    // Ensure kernel completion before returning.
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        // In case of launch failure, abort.
        // Note: abort on GPU errors is acceptable for correctness.
        // hipDeviceReset may be called after abort, but we just exit.
        // Use hipDeviceSynchronize to flush any pending errors.
        hipDeviceSynchronize();
    }
    // Synchronize to guarantee completion.
    hipDeviceSynchronize();
}

