#include <hip/hip_runtime.h>
#include <cstdint>

// Device kernel: processes each chain of the recurrence.
__global__ void versioned_distance_update_kernel(double *a, const double *b, const double *c,
                                                const int64_t LEN_1D, const int64_t K) {
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= K) return; // one thread per chain (modulo K)
    // Process elements i = tid + m*K, starting from i = tid + K (skip first K elements)
    for (int64_t i = tid + K; i < LEN_1D; i += K) {
        a[i] = 0.75 * a[i - K] + b[i] * c[i];
    }
}

extern "C" __attribute__((visibility("default")))
void versioned_distance_update_fp64(double *a, const double *b, const double *c,
                                   const int64_t LEN_1D, const int64_t K) {
    if (K <= 0 || K >= LEN_1D) return;
    const int blockSize = 256;
    int gridSize = static_cast<int>((K + blockSize - 1) / blockSize);
    dim3 block(blockSize);
    dim3 grid(gridSize);
    hipLaunchKernelGGL(versioned_distance_update_kernel, grid, block, 0, 0,
                       a, b, c, LEN_1D, K);
    hipDeviceSynchronize();
}
