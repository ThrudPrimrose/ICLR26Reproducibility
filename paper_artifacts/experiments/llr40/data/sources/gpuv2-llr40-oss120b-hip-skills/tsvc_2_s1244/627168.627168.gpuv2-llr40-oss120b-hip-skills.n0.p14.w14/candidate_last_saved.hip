#include <hip/hip_runtime.h>
#include <cstdint>

// Kernel to compute a[i] for i in [0, n-1]
extern "C" __global__ void tsvc_2_s1244_fp64_compute_a(double * __restrict__ a,
                                                       const double * __restrict__ b,
                                                       const double * __restrict__ c,
                                                       int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = b[i] + c[i] * c[i] + b[i] * b[i] + c[i];
    }
}

// Kernel to compute d[i] = a[i] + a[i+1] for i in [0, n-1]
extern "C" __global__ void tsvc_2_s1244_fp64_compute_d(const double * __restrict__ a,
                                                       double * __restrict__ d,
                                                       int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        d[i] = a[i] + a[i + 1];
    }
}

// Host wrapper matching the C ABI expected by the benchmark harness.
extern "C" void tsvc_2_s1244_fp64(double * __restrict__ a,
                                   const double * __restrict__ b,
                                   const double * __restrict__ c,
                                   double * __restrict__ d,
                                   const int64_t LEN_1D) {
    // Number of elements to process (LEN_1D-1)
    int64_t n = LEN_1D - 1;
    if (n <= 0) return;
    const int blockSize = 256;
    int64_t gridSize = (n + blockSize - 1) / blockSize;

    // Launch kernel to compute a
    hipLaunchKernelGGL(tsvc_2_s1244_fp64_compute_a,
                       dim3(gridSize), dim3(blockSize), 0, 0,
                       a, b, c, n);
    // Launch kernel to compute d using the newly computed a
    hipLaunchKernelGGL(tsvc_2_s1244_fp64_compute_d,
                       dim3(gridSize), dim3(blockSize), 0, 0,
                       a, d, n);
    // Synchronize before returning (optional, default stream ordering ensures correctness)
    hipDeviceSynchronize();
}
