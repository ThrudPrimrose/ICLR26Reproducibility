#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Device kernel that processes a single row ``i`` of the matrix.
// It updates a[i][j] += a[i-1][j] + a[i-1][j+1] for all valid columns j.
extern "C" __global__ void wf_diff_skew_fp64_row_kernel(double * __restrict__ a,
                                                       const int64_t LEN_2D,
                                                       const int64_t i) {
    // Compute column index j.
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < LEN_2D - 1) {
        size_t idx    = (size_t)i * (size_t)LEN_2D + (size_t)j;
        size_t idx_up = (size_t)(i - 1) * (size_t)LEN_2D + (size_t)j;
        // idx_up+1 is safe because j < LEN_2D-1.
        a[idx] = a[idx] + a[idx_up] + a[idx_up + 1];
    }
}

// Host entry point matching the benchmark's C ABI.
extern "C" void wf_diff_skew_fp64(double * __restrict__ a, const int64_t LEN_2D) {
    // Kernel launch configuration for processing a row.
    const int BLOCK_X = 256; // threads per block (adjust for occupancy)
    dim3 blockDim(BLOCK_X);
    dim3 gridDim(((size_t)LEN_2D - 1 + BLOCK_X - 1) / BLOCK_X);

    // Process rows sequentially to respect the data dependency on the previous row.
    for (int64_t i = 1; i < LEN_2D; ++i) {
        hipLaunchKernelGGL(wf_diff_skew_fp64_row_kernel,
                           gridDim, blockDim, 0, 0,
                           a, LEN_2D, i);
        // Check for launch errors.
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            fprintf(stderr, "hipLaunchKernelGGL failed at row %lld: %s\n", (long long)i, hipGetErrorString(err));
            fflush(stderr);
            exit(1);
        }
        // Ensure the kernel has completed before launching the next row.
        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            fprintf(stderr, "hipDeviceSynchronize failed at row %lld: %s\n", (long long)i, hipGetErrorString(err));
            fflush(stderr);
            exit(1);
        }
    }
}

