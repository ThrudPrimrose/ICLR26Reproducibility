#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

// Forward declaration of the device kernel (defined in device_source)
extern "C" __global__ void fuse_move_ifs_fp64_kernel(double* __restrict__ a,
                                                    double* __restrict__ b,
                                                    const double* __restrict__ cond,
                                                    const double* __restrict__ src,
                                                    int64_t K,
                                                    int64_t LEN_2D);

extern "C" void fuse_move_ifs_fp64(double* __restrict__ a,
                                   double* __restrict__ b,
                                   const double* __restrict__ cond,
                                   const double* __restrict__ src,
                                   const int64_t K,
                                   const int64_t LEN_2D) {
    const int64_t total = LEN_2D * LEN_2D;
    const int threads_per_block = 256;
    const int blocks = static_cast<int>((total + threads_per_block - 1) / threads_per_block);
    hipLaunchKernelGGL(fuse_move_ifs_fp64_kernel,
                       dim3(blocks), dim3(threads_per_block), 0, 0,
                       a, b, cond, src, K, LEN_2D);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "HIP error launching fuse_move_ifs_fp64_kernel: %s\n", hipGetErrorString(err));
        abort();
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "HIP error after synchronization: %s\n", hipGetErrorString(err));
        abort();
    }
}
