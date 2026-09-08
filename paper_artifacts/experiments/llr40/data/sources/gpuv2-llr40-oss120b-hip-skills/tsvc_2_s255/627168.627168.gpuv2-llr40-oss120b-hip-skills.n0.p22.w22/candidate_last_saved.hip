#include <hip/hip_runtime.h>
#include <cstdint>

// Device kernel: computes a[i] = (b[i] + b[i-1] + b[i-2]) * 0.333 with wrap-around.
extern "C" __global__ void tsvc_2_s255_fp64_kernel(double* __restrict__ a,
                                                   const double* __restrict__ b,
                                                   const int64_t LEN_1D) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < LEN_1D) {
        int64_t im1 = (i == 0) ? (LEN_1D - 1) : (i - 1);
        int64_t im2;
        if (i == 0) {
            im2 = LEN_1D - 2;
        } else if (i == 1) {
            im2 = LEN_1D - 1;
        } else {
            im2 = i - 2;
        }
        double sum = b[i] + b[im1] + b[im2];
        a[i] = sum * 0.333;
    }
}

extern "C" void tsvc_2_s255_fp64(double* __restrict__ a,
                                 const double* __restrict__ b,
                                 const int64_t LEN_1D) {
    // Choose a reasonable block size.
    const int blockSize = 256;
    int gridSize = static_cast<int>((LEN_1D + blockSize - 1) / blockSize);
    // Launch kernel.
    hipLaunchKernelGGL(tsvc_2_s255_fp64_kernel,
                       dim3(gridSize), dim3(blockSize), 0, 0,
                       a, b, LEN_1D);
    // Check for launch errors.
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        // Abort on error; the harness may catch stdout.
        fprintf(stderr, "HIP kernel launch failed: %s\n", hipGetErrorString(err));
        abort();
    }
    // Synchronize to ensure completion before returning.
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "HIP device synchronize failed: %s\n", hipGetErrorString(err));
        abort();
    }
}
