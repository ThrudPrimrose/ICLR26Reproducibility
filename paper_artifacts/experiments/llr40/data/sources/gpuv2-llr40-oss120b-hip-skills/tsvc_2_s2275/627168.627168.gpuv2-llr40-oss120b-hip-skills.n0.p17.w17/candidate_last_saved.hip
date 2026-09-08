// HIP implementation of tsvc_2_s2275 microkernel (fp64 version).
// Computes:
//   for i in 0..N-1:
//     for j in 0..N-1:
//       aa[i + j*N] += bb[i + j*N] * cc[i + j*N];
//     a[i] = b[i] + c[i] * d[i];
// Arguments are device pointers provided by the harness.

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Simple error-check macro.
#define HIP_CHECK(call) do { hipError_t err = (call); if (err != hipSuccess) { \
    fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
    std::exit(EXIT_FAILURE); } } while (0)

// Matrix kernel: each thread processes one element of the LEN_2D x LEN_2D matrix.
extern "C" __global__
void tsvc2_s2275_mat_kernel(double* __restrict aa,
                             const double* __restrict bb,
                             const double* __restrict cc,
                             int64_t N) {
    // Compute 2D indices.
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; // row index
    int64_t j = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y; // column index
    if (i < N && j < N) {
        // Column-major layout: idx = j * N + i
        int64_t idx = j * N + i;
        aa[idx] = aa[idx] + bb[idx] * cc[idx];
    }
}

// Vector kernel: compute a[i] = b[i] + c[i] * d[i]
extern "C" __global__
void tsvc2_s2275_vec_kernel(double* __restrict a,
                             const double* __restrict b,
                             const double* __restrict c,
                             const double* __restrict d,
                             int64_t N) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < N) {
        a[i] = b[i] + c[i] * d[i];
    }
}

// Exported entry point with C linkage matching the reference signature.
extern "C" void tsvc_2_s2275_fp64(double* __restrict a,
                                 double* __restrict aa,
                                 const double* __restrict b,
                                 const double* __restrict bb,
                                 const double* __restrict c,
                                 const double* __restrict cc,
                                 const double* __restrict d,
                                 const int64_t LEN_2D) {
    // Determine launch configuration for the matrix kernel.
    const int64_t N = LEN_2D;
    // Choose 16x16 thread blocks (256 threads). Adjust if N is small.
    const int tx = 16;
    const int ty = 16;
    dim3 blockSize(tx, ty);
    dim3 gridSize((N + tx - 1) / tx, (N + ty - 1) / ty);
    // Launch matrix kernel.
    hipLaunchKernelGGL(tsvc2_s2275_mat_kernel, gridSize, blockSize, 0, 0,
                       aa, bb, cc, N);
    // Check launch error.
    HIP_CHECK(hipGetLastError());

    // Vector kernel launch. Use 256 threads per block.
    const int vecBlock = 256;
    dim3 vecGrid((N + vecBlock - 1) / vecBlock);
    hipLaunchKernelGGL(tsvc2_s2275_vec_kernel, vecGrid, dim3(vecBlock), 0, 0,
                       a, b, c, d, N);
    HIP_CHECK(hipGetLastError());

    // Synchronize to ensure kernels have finished before returning.
    HIP_CHECK(hipDeviceSynchronize());
}

