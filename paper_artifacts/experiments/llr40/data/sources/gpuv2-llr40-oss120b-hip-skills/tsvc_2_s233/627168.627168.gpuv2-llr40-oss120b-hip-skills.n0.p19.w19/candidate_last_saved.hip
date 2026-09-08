// Host part of tsvc_2_s233 HIP implementation.
// Exposes the C ABI entry point required by the benchmark harness.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

// Forward declarations of device kernels (defined in device_source).
extern "C" __global__ void aa_kernel(double* __restrict__ aa,
                                      const double* __restrict__ cc,
                                      int64_t LEN_2D);
extern "C" __global__ void bb_kernel(double* __restrict__ bb,
                                      const double* __restrict__ cc,
                                      int64_t LEN_2D);

extern "C" void tsvc_2_s233_fp64(double* __restrict__ aa,
                                   double* __restrict__ bb,
                                   const double* __restrict__ cc,
                                   const int64_t LEN_2D) {
    const int threads = 256;
    int64_t work = LEN_2D - 8;
    if (work < 0) work = 0;
    int blocks = static_cast<int>((work + threads - 1) / threads);

    // Column scan for aa
    hipLaunchKernelGGL(aa_kernel,
                       dim3(blocks), dim3(threads), 0, 0,
                       aa, cc, LEN_2D);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "aa_kernel launch failed: %s\n", hipGetErrorString(err));
        abort();
    }

    // Row scan for bb
    hipLaunchKernelGGL(bb_kernel,
                       dim3(blocks), dim3(threads), 0, 0,
                       bb, cc, LEN_2D);
    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "bb_kernel launch failed: %s\n", hipGetErrorString(err));
        abort();
    }

    // Ensure kernels are complete before returning.
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "HIP sync failed: %s\n", hipGetErrorString(err));
        abort();
    }
}
