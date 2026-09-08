#include <hip/hip_runtime.h>
#include <cstdint>

extern "C" __global__ void tsvc_2_s235_kernel(double* a, double* aa, const double* b, const double* bb, const double* c, const int64_t LEN_2D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= LEN_2D) return;
    // compute a[i]
    double ai = a[i] + b[i] * c[i];
    a[i] = ai;
    // inner loop over j
    for (int64_t j = 1; j < LEN_2D; ++j) {
        // compute flat index for aa and bb
        int64_t idx = j * LEN_2D + i;
        int64_t idx_prev = (j - 1) * LEN_2D + i;
        aa[idx] = aa[idx_prev] + bb[idx] * ai;
    }
}

extern "C" void tsvc_2_s235_fp64(double* a, double* aa, const double* b, const double* bb, const double* c, const int64_t LEN_2D) {
    // launch kernel with enough threads to cover LEN_2D
    int threads = 256;
    int blocks = (LEN_2D + threads - 1) / threads;
    hipLaunchKernelGGL(tsvc_2_s235_kernel, dim3(blocks), dim3(threads), 0, 0, a, aa, b, bb, c, LEN_2D);
    // ensure completion
    hipDeviceSynchronize();
}

