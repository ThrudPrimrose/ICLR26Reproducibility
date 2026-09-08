#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

// Forward declaration of device kernel
__global__ void fuse_diamond_kernel(const double * __restrict__ a, double * __restrict__ out, const int64_t n);

extern "C" void fuse_diamond_fp64(const double * __restrict__ a, double * __restrict__ out, const int64_t LEN_1D) {
    // Determine launch configuration
    const int threadsPerBlock = 256;
    int blocks = (LEN_1D + threadsPerBlock - 1) / threadsPerBlock;
    // Launch kernel
    hipLaunchKernelGGL(fuse_diamond_kernel, dim3(blocks), dim3(threadsPerBlock), 0, 0, a, out, LEN_1D);
    // Synchronize to ensure completion before returning
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "HIP kernel launch failed: %s\n", hipGetErrorString(err));
        return;
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "HIP device sync failed: %s\n", hipGetErrorString(err));
        return;
    }
}

// Device kernel
__global__ void fuse_diamond_kernel(const double * __restrict__ a, double * __restrict__ out, const int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        double t = a[i] * a[i];
        out[i] = (t + 1.0) * (t - 1.0);
    }
}

