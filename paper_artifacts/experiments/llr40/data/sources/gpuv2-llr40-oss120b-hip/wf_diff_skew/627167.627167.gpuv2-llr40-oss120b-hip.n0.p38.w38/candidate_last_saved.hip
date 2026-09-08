#include <hip/hip_runtime.h>
#include <cstdint>

extern "C" __global__ void wf_diff_skew_kernel_fp64(double* __restrict__ a, const int64_t LEN_2D, const int64_t i) {
    int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < LEN_2D - 1) {
        // Compute linear indices
        int64_t idx = i * LEN_2D + j;
        int64_t idx_up = (i - 1) * LEN_2D + j;
        int64_t idx_up_right = (i - 1) * LEN_2D + (j + 1);
        a[idx] = a[idx] + a[idx_up] + a[idx_up_right];
    }
}

extern "C" void wf_diff_skew_fp64(double* __restrict__ a, const int64_t LEN_2D) {
    // Number of columns to process per row (excluding last column)
    const int64_t ncols = LEN_2D - 1;
    const int blockSize = 256; // tune as needed
    const int64_t gridSize = (ncols + blockSize - 1) / blockSize;
    dim3 block(blockSize);
    dim3 grid(gridSize);
    for (int64_t i = 1; i < LEN_2D; ++i) {
        // Launch kernel for this row
        hipLaunchKernelGGL(wf_diff_skew_kernel_fp64, grid, block, 0, 0, a, LEN_2D, i);
        // Optional: check for launch errors
        // hipError_t err = hipGetLastError();
        // if (err != hipSuccess) return; // or abort
    }
    // Ensure all work is finished before returning
    hipDeviceSynchronize();
}
