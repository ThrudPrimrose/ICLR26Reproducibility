#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

#define HIP_CHECK(call)                                                 \
    do {                                                                \
        hipError_t err = call;                                          \
        if (err != hipSuccess) {                                        \
            fprintf(stderr, "HIP error %s at %s:%d\n",                \
                    hipGetErrorString(err), __FILE__, __LINE__);      \
            exit(1);                                                   \
        }                                                               \
    } while (0)

// Device kernel that processes a single row "row" of the matrix.
// It updates aa[row, j] = aa[row-1, j-1] + bb[row, j] for all j >= 1.
// N = LEN_2D, total dimension of the square matrix.

__global__ void tsvc_2_s119_row_kernel(double* __restrict__ aa,
                                        const double* __restrict__ bb,
                                        int N,
                                        int row) {
    // Global column index, starting from 1 (the first column to update).
    int j = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N) {
        int idx = row * N + j;                // index for aa[row, j] and bb[row, j]
        int idx_prev = (row - 1) * N + (j - 1); // index for aa[row-1, j-1]
        aa[idx] = aa[idx_prev] + bb[idx];
    }
}

// Exported C-ABI entry point expected by the benchmark harness.
extern "C" void tsvc_2_s119_fp64(double* __restrict__ aa,
                                 const double* __restrict__ bb,
                                 const int64_t LEN_2D) {
    int N = static_cast<int>(LEN_2D);
    // Guard against degenerate sizes.
    if (N <= 1) return;

    const int blockSize = 256; // reasonable launch size, works for any LEN_2D
    int numElements = N - 1;   // number of columns to update per row (j = 1..N-1)
    int gridSize = (numElements + blockSize - 1) / blockSize;

    // Process rows sequentially to respect the wave‑front dependency.
    for (int row = 1; row < N; ++row) {
        // Launch one row kernel.
        hipLaunchKernelGGL(tsvc_2_s119_row_kernel,
                           dim3(gridSize), dim3(blockSize), 0, 0,
                           aa, bb, N, row);
        // Check launch and synchronize before the next row.
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
    }
}

