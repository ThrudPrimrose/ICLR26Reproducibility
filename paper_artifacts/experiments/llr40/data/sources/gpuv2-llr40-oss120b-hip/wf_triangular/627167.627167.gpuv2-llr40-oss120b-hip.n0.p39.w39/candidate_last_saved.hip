#include <hip/hip_runtime.h>
#include <cstdint>

// Kernel that processes a single diagonal d = i + j.
extern "C" __global__ void wf_triangular_kernel_fp64(double* __restrict__ a, const int64_t LEN_2D, const int64_t d) {
    // Compute the valid range of i for this diagonal.
    int64_t i_min = (1 > d - (LEN_2D - 1)) ? 1 : d - (LEN_2D - 1);
    int64_t i_max = ((LEN_2D - 1) < (d / 2)) ? (LEN_2D - 1) : (d / 2);
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t i = i_min + idx;
    if (i > i_max) return;
    int64_t j = d - i;
    // Linear indices for the current element and its dependencies.
    int64_t pos   = i * LEN_2D + j;
    int64_t up    = (i - 1) * LEN_2D + j;
    int64_t left  = i * LEN_2D + (j - 1);
    a[pos] = a[pos] + a[up] + a[left];
}

// Host entry point matching the C ABI expected by the benchmark.
extern "C" void wf_triangular_fp64(double* __restrict__ a, const int64_t LEN_2D) {
    const int blockSize = 256; // Tune if needed.
    // The maximum diagonal index for i,j in [1, LEN_2D-1] is 2*(LEN_2D-1).
    const int64_t maxDiag = 2 * (LEN_2D - 1);
    for (int64_t d = 2; d <= maxDiag; ++d) {
        // Compute number of elements on this diagonal.
        int64_t i_min = (1 > d - (LEN_2D - 1)) ? 1 : d - (LEN_2D - 1);
        int64_t i_max = ((LEN_2D - 1) < (d / 2)) ? (LEN_2D - 1) : (d / 2);
        int64_t count = i_max >= i_min ? (i_max - i_min + 1) : 0;
        if (count == 0) continue;
        int64_t gridSize = (count + blockSize - 1) / blockSize;
        dim3 block(blockSize);
        dim3 grid(gridSize);
        hipLaunchKernelGGL(wf_triangular_kernel_fp64, grid, block, 0, 0, a, LEN_2D, d);
        // Ensure the diagonal finishes before the next one begins.
        hipDeviceSynchronize();
    }
    // Final synchronization before returning.
    hipDeviceSynchronize();
}

