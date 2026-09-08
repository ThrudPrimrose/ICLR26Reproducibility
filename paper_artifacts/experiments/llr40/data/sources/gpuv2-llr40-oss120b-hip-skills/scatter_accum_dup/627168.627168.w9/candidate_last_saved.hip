#include <hip/hip_runtime.h>
#include <cstdint>

// Device kernel: parallel scatter-accumulate using atomicAdd for double precision.
__global__ void scatter_accum_dup_kernel(double* __restrict__ bins, const double* __restrict__ src,
                                         const int* __restrict__ ip, int64_t LEN_1D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Grid-stride loop to cover all elements.
    for (int64_t i = idx; i < LEN_1D; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        // Each iteration may write to a duplicate index, so use atomicAdd.
        // ip[i] is guaranteed to be within the bounds of bins.
        atomicAdd(&bins[ip[i]], src[i]);
    }
}

extern "C" void scatter_accum_dup_fp64(double* __restrict__ bins,
                                        const double* __restrict__ src,
                                        const int* __restrict__ ip,
                                        const int64_t LEN_1D) {
    // Choose a reasonable block size.
    const int blockSize = 256;
    // Compute grid size to cover all elements.
    int64_t gridSize = (LEN_1D + blockSize - 1) / blockSize;
    if (gridSize > 65535) {
        gridSize = 65535; // limit for 1D grid dimension.
    }
    // Launch kernel.
    hipLaunchKernelGGL(scatter_accum_dup_kernel, dim3(static_cast<unsigned int>(gridSize)),
                       dim3(blockSize), 0, 0,
                       bins, src, ip, LEN_1D);
    // Ensure kernel completion before returning.
    hipDeviceSynchronize();
}
