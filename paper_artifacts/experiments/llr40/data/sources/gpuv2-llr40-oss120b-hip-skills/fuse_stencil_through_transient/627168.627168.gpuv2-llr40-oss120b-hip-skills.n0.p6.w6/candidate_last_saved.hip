#include <hip/hip_runtime.h>
#include <cstdint>

// Device kernel performing the fused stencil computation.
// Computes out[i] = (a[i-1] + a[i] + a[i+1]) * (a[i] + a[i+1] + a[i+2])
// for i in [1, LEN_1D - 3].
extern "C" __global__ void fuse_stencil_kernel(const double* __restrict__ a,
                                                double* __restrict__ out,
                                                const int64_t LEN_1D) {
    // Compute global index starting from 1.
    int64_t i = 1 + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const int64_t last = LEN_1D - 3; // inclusive upper bound.
    for (; i <= last; i += stride) {
        double sum1 = a[i - 1] + a[i] + a[i + 1];
        double sum2 = a[i] + a[i + 1] + a[i + 2];
        out[i] = sum1 * sum2;
    }
}

// Host entry point matching the C ABI used by the reference implementation.
extern "C" void fuse_stencil_through_transient_fp64(const double* __restrict__ a,
                                                    double* __restrict__ out,
                                                    const int64_t LEN_1D) {
    // Choose a reasonable launch configuration. 256 threads per block is a typical default.
    const int threads = 256;
    // Number of blocks to cover the range; we can over-provision because the kernel guards bounds.
    int blocks = (static_cast<int>(LEN_1D) + threads - 1) / threads;
    // Ensure at least one block; also cap to a reasonable maximum to avoid launch failures on tiny inputs.
    if (blocks < 1) blocks = 1;

    // Launch the kernel. The stream argument 0 selects the default stream.
    hipLaunchKernelGGL(fuse_stencil_kernel,
                       dim3(blocks), dim3(threads), 0, 0,
                       a, out, LEN_1D);
    // Wait for kernel completion before returning to the harness.
    hipDeviceSynchronize();
}

