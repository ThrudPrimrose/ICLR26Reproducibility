#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

// Kernel that computes b[i] = d[i] * e[i]; then a[i] += b[i] * c[i]
__global__ void tsvc_2_s152_kernel(double* __restrict__ a,
                                   double* __restrict__ b,
                                   const double* __restrict__ c,
                                   const double* __restrict__ d,
                                   const double* __restrict__ e,
                                   int64_t N) {
    int64_t gid = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    int64_t stride = static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
    for (int64_t i = gid; i < N; i += stride) {
        double t = d[i] * e[i];
        b[i] = t;
        a[i] += t * c[i];
    }
}

extern "C" void tsvc_2_s152_fp64(double* __restrict__ a,
                                 double* __restrict__ b,
                                 const double* __restrict__ c,
                                 const double* __restrict__ d,
                                 const double* __restrict__ e,
                                 int64_t LEN_1D) {
    // Determine launch configuration.
    const int blockSize = 256; // typical warp size multiple
    // Compute grid size, but cap to a reasonable maximum.
    int maxGrid = 65535; // safe upper bound for most GPUs
    int64_t gridCalc = (LEN_1D + blockSize - 1) / blockSize;
    int gridSize = static_cast<int>(gridCalc > maxGrid ? maxGrid : gridCalc);

    // Launch kernel.
    hipLaunchKernelGGL(tsvc_2_s152_kernel,
                       dim3(gridSize), dim3(blockSize), 0, 0,
                       a, b, c, d, e, LEN_1D);
    // Check launch errors.
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "HIP launch error: %s\n", hipGetErrorString(err));
        // Abort to signal failure.
        abort();
    }
    // Synchronize to ensure completion before returning.
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "HIP sync error: %s\n", hipGetErrorString(err));
        abort();
    }
}
