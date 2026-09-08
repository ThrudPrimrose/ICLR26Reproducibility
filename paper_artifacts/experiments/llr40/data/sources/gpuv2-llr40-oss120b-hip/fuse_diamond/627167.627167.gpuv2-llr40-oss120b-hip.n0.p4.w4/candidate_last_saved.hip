#include <hip/hip_runtime.h>
#include <cstdint>

extern "C" __global__ void fuse_diamond_kernel_fp64(const double* __restrict__ a, double* __restrict__ out, int64_t LEN) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < LEN) {
        double ai = a[idx];
        double ai2 = ai * ai;
        out[idx] = ai2 * ai2 - 1.0;
    }
}

extern "C" void fuse_diamond_fp64(const double* __restrict__ a, double* __restrict__ out, const int64_t LEN_1D) {
    const int blockSize = 256;
    int gridSize = (LEN_1D + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(fuse_diamond_kernel_fp64, dim3(gridSize), dim3(blockSize), 0, 0, a, out, LEN_1D);
    hipDeviceSynchronize();
}

