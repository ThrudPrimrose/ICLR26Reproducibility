#include <hip/hip_runtime.h>
#include <cstdint>

// HIP kernel: compute a[i] = (b[i] + b[i-1] + b[i-2]) * 0.333, with wrap-around.
extern "C" __global__ void tsvc_2_s255_kernel(double* __restrict__ a,
                                             const double* __restrict__ b,
                                             const int64_t LEN_1D) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= LEN_1D) return;
    // Compute indices for b[i-1] and b[i-2] with wrap-around.
    int64_t i1 = (i == 0) ? (LEN_1D - 1) : (i - 1);
    int64_t i2 = (i <= 1) ? (LEN_1D - (2 - i)) : (i - 2);
    double val = (b[i] + b[i1] + b[i2]) * 0.333;
    a[i] = val;
}

extern "C" void tsvc_2_s255_fp64(double* __restrict__ a,
                                 const double* __restrict__ b,
                                 const int64_t LEN_1D) {
    const int blockSize = 256;
    int gridSize = (LEN_1D + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(tsvc_2_s255_kernel,
                       dim3(gridSize), dim3(blockSize), 0, 0,
                       a, b, LEN_1D);
    // No need to synchronize; the caller will handle any necessary sync.
}
