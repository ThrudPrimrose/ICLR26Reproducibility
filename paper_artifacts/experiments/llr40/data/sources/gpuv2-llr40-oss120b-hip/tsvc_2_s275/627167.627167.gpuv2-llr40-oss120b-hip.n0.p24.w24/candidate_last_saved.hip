#include <hip/hip_runtime.h>
#include <cstdint>

// Device kernel: processes one column per thread
__global__ void tsvc_2_s275_kernel(double *aa, const double *bb, const double *cc, int64_t LEN) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= LEN) return;

    // Load the first element of column i
    double a0 = aa[i]; // corresponds to aa[0, i]
    if (a0 > 0.0) {
        double cur = a0;
        // Process rows j = 1 .. LEN-1
        // Compute linear index: row * LEN + column i
        for (int64_t j = 1; j < LEN; ++j) {
            int64_t idx = j * LEN + i;
            double b = bb[idx];
            double c = cc[idx];
            cur = cur + b * c;
            aa[idx] = cur;
        }
    }
    // else do nothing (aa[0,i] unchanged)
}

extern "C" void tsvc_2_s275_fp64(double *aa, const double *bb, const double *cc, const int64_t LEN_2D) {
    // Launch one thread per column
    const int threads_per_block = 256;
    int blocks = (LEN_2D + threads_per_block - 1) / threads_per_block;
    hipLaunchKernelGGL(tsvc_2_s275_kernel, dim3(blocks), dim3(threads_per_block), 0, 0,
                       aa, bb, cc, LEN_2D);
    // Ensure kernel completion before returning (optional, but to match semantics)
    hipDeviceSynchronize();
}

