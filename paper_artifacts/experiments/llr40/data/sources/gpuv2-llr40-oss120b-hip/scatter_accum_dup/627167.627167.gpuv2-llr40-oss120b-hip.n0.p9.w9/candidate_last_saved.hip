#include <hip/hip_runtime.h>

// Forward declaration of device kernels.
__global__ void scatter_accum_dup_kernel(double *bins, const double *src, const int *ip, int N);
__global__ void scatter_accum_dup_kernel_noatomic(double *bins, const double *src, const int *ip, int N);

extern "C" void scatter_accum_dup_fp64(double *bins, int32_t *ip, double *src, int64_t LEN_1D, uint8_t *workspace, int64_t workspace_bytes) {
    int threads = 256;
    int blocks = (LEN_1D + threads - 1) / threads;
    // Launch atomic kernel by default - detection logic to be added later.
    hipLaunchKernelGGL(scatter_accum_dup_kernel,
                       dim3(blocks), dim3(threads), 0, 0,
                       bins, src, ip, LEN_1D);
    hipDeviceSynchronize();
}
