/*
 * Host entry point for the ``segment_reduce_ragged`` kernel.
 * The actual device kernel lives in `segment_reduce_ragged_device.hip`.
 */

#include "segment_reduce_ragged_device.hip"

#if defined(__HIPCC__)
#include "/opt/rocm/include/hip/hip_runtime.h"
#else
#include <cstdint>
#define HIP_KERNEL_NAME(x) x
#define hipLaunchKernelGGL(kernelName, gridDim, blockDim, sharedMem, stream, ...) ((void)0)
#define __global__
#endif
#include <cstdint>

// Forward declaration of the device kernel (exact signature must match the device file).
extern "C" __global__ void segment_reduce_ragged_kernel(const int64_t * __restrict__ row_ptr,
                                                       const double * __restrict__ val,
                                                       const double * __restrict__ w,
                                                       double * __restrict__ out,
                                                       const int64_t NSEG);

extern "C" void segment_reduce_ragged_fp64(const int64_t * __restrict__ row_ptr,
                                            const double * __restrict__ val,
                                            const double * __restrict__ w,
                                            double * __restrict__ out,
                                            const int64_t NSEG) {
    // Choose a block size that is a multiple of the warp size. 256 threads (8 warps)
    // provides good occupancy on most GPUs.
    constexpr int blockSize = 256;               // must be divisible by 32 (warp size)
    constexpr int warpsPerBlock = blockSize / 32;

    // Compute the number of warps required to cover all segments.
    int64_t requiredWarps = (NSEG + warpsPerBlock - 1) / warpsPerBlock;
    // Convert to a grid size (number of blocks).  Ensure we round up.
    int gridSize = static_cast<int>((requiredWarps + warpsPerBlock - 1) / warpsPerBlock);

    // Launch the device kernel.
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(segment_reduce_ragged_kernel),
        dim3(gridSize), dim3(blockSize), 0, 0,
        row_ptr, val, w, out, NSEG);
}

