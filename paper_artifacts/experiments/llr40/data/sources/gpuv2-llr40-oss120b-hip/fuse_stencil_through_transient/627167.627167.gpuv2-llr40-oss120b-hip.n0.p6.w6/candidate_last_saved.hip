#include <hip/hip_runtime.h>
#include <cstdint>

// Forward declaration of the device kernel defined in the device source
extern "C" __global__ void fuse_stencil_through_transient_kernel(const double* __restrict__ a,
                                                                double* __restrict__ out,
                                                                int64_t LEN_1D);

extern "C" void fuse_stencil_through_transient_fp64(const double* __restrict__ a,
                                                   double* __restrict__ out,
                                                   const int64_t LEN_1D) {
    // Number of iterations: i from 1 to LEN_1D-3 inclusive
    int64_t n = LEN_1D - 3;
    if (n <= 0) {
        return; // nothing to compute
    }
    const int blockSize = 256;
    int64_t gridSize = (n + blockSize - 1) / blockSize;
    // Cap grid size to the maximum allowed by the launch API (65535 on many GPUs)
    if (gridSize > 65535) gridSize = 65535;
    hipLaunchKernelGGL(fuse_stencil_through_transient_kernel,
                       dim3(gridSize), dim3(blockSize), 0, 0,
                       a, out, LEN_1D);
    hipDeviceSynchronize();
}
