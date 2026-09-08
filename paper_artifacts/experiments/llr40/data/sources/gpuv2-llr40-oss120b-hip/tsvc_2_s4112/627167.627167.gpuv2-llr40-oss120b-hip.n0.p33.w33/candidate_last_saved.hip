#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

// Device kernel
__global__ void tsvc_2_s4112_kernel(double* __restrict__ a,
                                   const double* __restrict__ b,
                                   const int32_t* __restrict__ ip,
                                   const int64_t LEN_1D) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < LEN_1D) {
        int64_t srcIdx = static_cast<int64_t>(ip[idx]);
        // Use __ldg to read read-only data from b (may improve cache usage)
        double bv = __ldg(&b[srcIdx]);
        a[idx] = a[idx] + bv * 2.0;
    }
}

extern "C" void tsvc_2_s4112_fp64(double* __restrict__ a,
                                 const double* __restrict__ b,
                                 const int32_t* __restrict__ ip,
                                 const int64_t LEN_1D) {
    const int blockSize = 256;
    int gridSize = static_cast<int>((LEN_1D + blockSize - 1) / blockSize);
    hipLaunchKernelGGL(tsvc_2_s4112_kernel,
                       dim3(gridSize), dim3(blockSize), 0, 0,
                       a, b, ip, LEN_1D);
    // Optional: check for launch errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        // In case of panic, abort.
        fprintf(stderr, "HIP kernel launch error: %s\n", hipGetErrorString(err));
    }
}
