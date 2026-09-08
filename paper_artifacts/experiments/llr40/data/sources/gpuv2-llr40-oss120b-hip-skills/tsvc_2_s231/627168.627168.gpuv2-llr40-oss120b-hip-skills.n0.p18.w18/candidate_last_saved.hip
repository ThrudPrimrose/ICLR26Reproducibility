#include <hip/hip_runtime.h>
#include <cstdint>

extern "C" __global__ void tsvc_2_s231_kernel(double * __restrict__ aa,
                                                const double * __restrict__ bb,
                                                const int64_t LEN_2D) {
    // Each thread processes one column i
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= LEN_2D) return;
    // Load the initial element (j = 0)
    double prev = aa[i]; // aa[0 * LEN_2D + i]
    // Loop over rows j = 1 .. LEN_2D-1
    for (int64_t j = 1; j < LEN_2D; ++j) {
        double val = prev + bb[j * LEN_2D + i];
        aa[j * LEN_2D + i] = val;
        prev = val;
    }
}

extern "C" void tsvc_2_s231_fp64(double * __restrict__ aa,
                                 const double * __restrict__ bb,
                                 const int64_t LEN_2D) {
    // Choose launch configuration
    const int blockSize = 256;
    int gridSize = static_cast<int>((LEN_2D + blockSize - 1) / blockSize);
    // Launch kernel
    hipLaunchKernelGGL(tsvc_2_s231_kernel,
                       dim3(gridSize), dim3(blockSize), 0, 0,
                       aa, bb, LEN_2D);
    // Synchronize to catch launch errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        // Abort on error
        fprintf(stderr, "HIP launch error: %s\n", hipGetErrorString(err));
        std::abort();
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "HIP sync error: %s\n", hipGetErrorString(err));
        std::abort();
    }
}

