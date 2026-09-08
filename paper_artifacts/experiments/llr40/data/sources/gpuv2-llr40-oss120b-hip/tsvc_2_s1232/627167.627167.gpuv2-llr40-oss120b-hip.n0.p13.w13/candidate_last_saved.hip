// Host wrapper for TSVC tsvc_2 s1232 kernel (fp64) on HIP.
// Provides the C‑ABI entry point expected by the benchmark harness.

#include <cstdint>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#ifndef __HIPCC__
// When compiled without HIP, __global__ is not defined; define it as empty.
#define __global__
#endif

// Forward declaration of the device kernel. It will be linked from the device source.
extern "C" __global__ void tsvc_2_s1232_kernel(double* __restrict__ aa,
                                           const double* __restrict__ bb,
                                           const double* __restrict__ cc,
                                           int64_t LEN_2D,
                                           int64_t VLEN);

extern "C" void tsvc_2_s1232_fp64(double* __restrict__ aa,
                                   const double* __restrict__ bb,
                                   const double* __restrict__ cc,
                                   const int64_t LEN_2D,
                                   const int64_t VLEN) {
#if defined(__HIPCC__)
    // Block size of 256 threads (32x8).
    const dim3 blockDim(32, 8);
    const dim3 gridDim((static_cast<int64_t>(LEN_2D) + blockDim.x - 1) / blockDim.x,
                       (static_cast<int64_t>(LEN_2D) + blockDim.y - 1) / blockDim.y);
    // Launch the kernel.
    hipLaunchKernelGGL(tsvc_2_s1232_kernel,
                       gridDim, blockDim,
                       0, 0,
                       aa, bb, cc, LEN_2D, VLEN);
    // Synchronize to ensure the kernel completes before returning.
    hipDeviceSynchronize();
#else
    // Fallback CPU implementation for syntax checking or non‑HIP builds.
    // Compute the same region as the GPU kernel.
    for (int64_t j = 0; j < LEN_2D; ++j) {
        int64_t i_start = j * VLEN;
        if (i_start >= LEN_2D) continue;
        for (int64_t i = i_start; i < LEN_2D; ++i) {
            int64_t idx = i * LEN_2D + j;
            aa[idx] = bb[idx] + cc[idx];
        }
    }
#endif
}
