#include <hip/hip_runtime.h>
#include <cstdint>

// Kernel: each thread processes a column (fixed i) sequentially over rows (j)
__global__ void tsvc_2_s231_kernel(double *aa, const double *bb, int64_t LEN_2D) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= LEN_2D) return;
    // Compute prefix sum down column i
    for (int64_t j = 1; j < LEN_2D; ++j) {
        aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + bb[j * LEN_2D + i];
    }
}

extern "C" void tsvc_2_s231_fp64(double *aa, const double *bb, int64_t LEN_2D) {
    const int blockSize = 256;
    int64_t gridSize = (LEN_2D + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(tsvc_2_s231_kernel, dim3(gridSize), dim3(blockSize), 0, 0, aa, bb, LEN_2D);
    hipDeviceSynchronize();
}
