#include <hip/hip_runtime.h>
#include <cstdint>

// Forward declaration of device kernel (defined in a separate source).
extern "C" __global__ void quasi_affine_reduce_odd_kernel(const double* __restrict__ a,
                                                          double* __restrict__ out,
                                                          int64_t LEN_1D);

extern "C" void quasi_affine_reduce_odd_fp64(const double* __restrict__ a,
                                             double* __restrict__ out,
                                             const int64_t LEN_1D) {
    if (LEN_1D <= 1) {
        hipMemset(out, 0, sizeof(double));
        hipDeviceSynchronize();
        return;
    }
    hipMemset(out, 0, sizeof(double));
    constexpr int BLOCK_SIZE = 256;
    int64_t numBlocks = (LEN_1D + (int64_t)BLOCK_SIZE * 2 - 1) / ((int64_t)BLOCK_SIZE * 2);
    if (numBlocks < 1) numBlocks = 1;
    dim3 grid((unsigned int)numBlocks);
    dim3 block(BLOCK_SIZE);
    hipLaunchKernelGGL(quasi_affine_reduce_odd_kernel, grid, block, 0, 0,
                       a, out, LEN_1D);
    hipDeviceSynchronize();
}
