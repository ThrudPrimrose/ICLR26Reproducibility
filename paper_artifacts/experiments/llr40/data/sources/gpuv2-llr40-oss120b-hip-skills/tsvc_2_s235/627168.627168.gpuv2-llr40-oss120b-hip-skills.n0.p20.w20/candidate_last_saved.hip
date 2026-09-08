#include <hip/hip_runtime.h>
#include <cstdint>

extern "C" __global__ void tsvc_2_s235_kernel(double* __restrict__ a, double* __restrict__ aa,
                                               const double* __restrict__ b, const double* __restrict__ bb,
                                               const double* __restrict__ c, const int64_t LEN_2D) {
    // Compute global thread index for i dimension
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= LEN_2D) return;
    // Update a[i]
    a[i] += b[i] * c[i];
    double a_val = a[i];
    // Compute aa[j][i] for all j >= 1
    // Use simple sequential loop; each iteration depends on previous aa.
    // Unroll a bit for performance.
    for (int64_t j = 1; j < LEN_2D; ++j) {
        aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + bb[j * LEN_2D + i] * a_val;
    }
}

extern "C" void tsvc_2_s235_fp64(double* __restrict__ a, double* __restrict__ aa,
                                 const double* __restrict__ b, const double* __restrict__ bb,
                                 const double* __restrict__ c, const int64_t LEN_2D) {
    // Choose a reasonable block size, e.g., 256 threads per block
    const int threadsPerBlock = 256;
    int blocks = static_cast<int>((LEN_2D + threadsPerBlock - 1) / threadsPerBlock);
    // Launch kernel
    hipLaunchKernelGGL(tsvc_2_s235_kernel, dim3(blocks), dim3(threadsPerBlock), 0, 0,
                       a, aa, b, bb, c, LEN_2D);
    // Wait for kernel completion
    hipDeviceSynchronize();
}

