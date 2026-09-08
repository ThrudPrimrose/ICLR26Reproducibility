#include <hip/hip_runtime.h>
#include <cstdint>

// Parallel block-level prefix sum (inclusive scan) that matches the sequential CPU
// result exactly. Each block processes a contiguous chunk of the array with a
// single thread, preserving the order of floating‑point additions.

// Compute the sum of each block (partial sum).
extern "C" __global__ void block_sum_kernel(const double * __restrict__ a,
                                            double * __restrict__ block_sums,
                                            int64_t N,
                                            int64_t BLOCK_SIZE) {
    int64_t blockId = blockIdx.x;
    int64_t start = blockId * BLOCK_SIZE;
    int64_t end = start + BLOCK_SIZE;
    if (end > N) end = N;
    double sum = 0.0;
    for (int64_t i = start; i < end; ++i) {
        sum += a[i];
    }
    block_sums[blockId] = sum;
}

// Scan each block using the offset obtained from the previous blocks.
extern "C" __global__ void block_scan_kernel(const double * __restrict__ a,
                                             double * __restrict__ b,
                                             const double * __restrict__ offsets,
                                             int64_t N,
                                             int64_t BLOCK_SIZE) {
    int64_t blockId = blockIdx.x;
    int64_t start = blockId * BLOCK_SIZE;
    int64_t end = start + BLOCK_SIZE;
    if (end > N) end = N;
    double sum = offsets[blockId]; // sum of all previous elements
    for (int64_t i = start; i < end; ++i) {
        sum += a[i];
        b[i] = sum;
    }
}

// Exported entry point expected by the benchmark harness.
extern "C" void tsvc_2_s3112_fp64(const double * __restrict__ a,
                                   double * __restrict__ b,
                                   const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;
    const int64_t BLOCK_SIZE = 256; // tuned for good occupancy
    int64_t numBlocks = (LEN_1D + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate temporary storage for per‑block sums and offsets.
    double *d_block_sums = nullptr;
    double *d_offsets = nullptr;
    hipMalloc(&d_block_sums, numBlocks * sizeof(double));
    // First pass: compute sum of each block.
    hipLaunchKernelGGL(block_sum_kernel,
                       dim3(numBlocks), dim3(1), 0, 0,
                       a, d_block_sums, LEN_1D, BLOCK_SIZE);
    hipDeviceSynchronize();

    // Copy block sums to host and compute exclusive prefix (offsets).
    double *h_block_sums = (double *)malloc(numBlocks * sizeof(double));
    hipMemcpy(h_block_sums, d_block_sums, numBlocks * sizeof(double), hipMemcpyDeviceToHost);
    double *h_offsets = (double *)malloc(numBlocks * sizeof(double));
    double run = 0.0;
    for (int64_t i = 0; i < numBlocks; ++i) {
        h_offsets[i] = run;
        run += h_block_sums[i];
    }
    // Transfer offsets back to the device.
    hipMalloc(&d_offsets, numBlocks * sizeof(double));
    hipMemcpy(d_offsets, h_offsets, numBlocks * sizeof(double), hipMemcpyHostToDevice);
    // Second pass: scan each block using its offset.
    hipLaunchKernelGGL(block_scan_kernel,
                       dim3(numBlocks), dim3(1), 0, 0,
                       a, b, d_offsets, LEN_1D, BLOCK_SIZE);
    hipDeviceSynchronize();
    // Clean up.
    hipFree(d_block_sums);
    hipFree(d_offsets);
    free(h_block_sums);
    free(h_offsets);
}
