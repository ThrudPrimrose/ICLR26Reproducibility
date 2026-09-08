#include <hip/hip_runtime.h>
#include <cstdint>

// Forward declaration of the kernel
__global__ void tsvc_2_s1232_kernel(double * __restrict__ aa,
                                    const double * __restrict__ bb,
                                    const double * __restrict__ cc,
                                    const int64_t LEN_2D,
                                    const int64_t VLEN);


extern "C" void tsvc_2_s1232_fp64(double * __restrict__ aa,
                                    const double * __restrict__ bb,
                                    const double * __restrict__ cc,
                                    const int64_t LEN_2D,
                                    const int64_t VLEN) {
    // Compute total number of elements
    const int64_t total = LEN_2D * LEN_2D;
    const int threadsPerBlock = 256;
    int64_t blocks = (total + threadsPerBlock - 1) / threadsPerBlock;
    // Launch kernel
    hipLaunchKernelGGL(
        tsvc_2_s1232_kernel,
        dim3(blocks), dim3(threadsPerBlock), 0, 0,
        aa, bb, cc, LEN_2D, VLEN);
    // Synchronize to ensure completion before returning
    hipDeviceSynchronize();
}

// Device kernel
__global__ void tsvc_2_s1232_kernel(double * __restrict__ aa,
                                    const double * __restrict__ bb,
                                    const double * __restrict__ cc,
                                    const int64_t LEN_2D,
                                    const int64_t VLEN) {
    int64_t gid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = LEN_2D * LEN_2D;
    if (gid >= total) return;
    int64_t i = gid / LEN_2D;
    int64_t j = gid % LEN_2D;
    if (i >= j * VLEN) {
        aa[gid] = bb[gid] + cc[gid];
    }
    // else leave aa unchanged
}
