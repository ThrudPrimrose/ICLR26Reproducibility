// Host part of the TSVC 2 s2233 HIP implementation.
// Exposes the C ABI entry point required by the benchmark harness.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

// Forward declarations of device kernels (defined in device_source).
extern "C" __global__ void scan_aa_kernel(double* __restrict__ aa,
                                          const double* __restrict__ cc,
                                          int64_t LEN_2D);
extern "C" __global__ void scan_bb_kernel(double* __restrict__ bb,
                                          const double* __restrict__ cc,
                                          int64_t LEN_2D);

extern "C" void tsvc_2_s2233_fp64(double* __restrict__ aa,
                                   double* __restrict__ bb,
                                   const double* __restrict__ cc,
                                   const int64_t LEN_2D) {
    const int threads = 256;
    int64_t work = LEN_2D - 8;
    if (work < 0) work = 0;
    int blocks = static_cast<int>((work + threads - 1) / threads);

    // Column scan for aa
    hipLaunchKernelGGL(scan_aa_kernel,
                       dim3(blocks), dim3(threads), 0, 0,
                       aa, cc, LEN_2D);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "HIP error launching scan_aa_kernel: %s\n", hipGetErrorString(err));
        abort();
    }

    // Row scan for bb
    hipLaunchKernelGGL(scan_bb_kernel,
                       dim3(blocks), dim3(threads), 0, 0,
                       bb, cc, LEN_2D);
    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "HIP error launching scan_bb_kernel: %s\n", hipGetErrorString(err));
        abort();
    }

    // Synchronize to ensure completion before returning.
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "HIP error after synchronization: %s\n", hipGetErrorString(err));
        abort();
    }
}
