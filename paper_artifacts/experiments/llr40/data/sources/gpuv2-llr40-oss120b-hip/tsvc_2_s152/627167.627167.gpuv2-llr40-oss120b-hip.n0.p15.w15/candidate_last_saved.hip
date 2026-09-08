#include <hip/hip_runtime.h>
#include <cstdint>

// Kernel: compute b[i] = d[i] * e[i]; a[i] += b[i] * c[i];
__global__ void tsvc_2_s152_fp64_kernel(double *a, double *b, const double *c, const double *d, const double *e, int64_t LEN_1D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < LEN_1D) {
        double tmp = d[idx] * e[idx];
        b[idx] = tmp;
        a[idx] += tmp * c[idx];
    }
}

extern "C" void tsvc_2_s152_fp64(double * __restrict__ a, double * __restrict__ b, const double * __restrict__ c,
                                 const double * __restrict__ d, const double * __restrict__ e, const int64_t LEN_1D) {
    const int threadsPerBlock = 256;
    int blocks = static_cast<int>((LEN_1D + threadsPerBlock - 1) / threadsPerBlock);
    hipLaunchKernelGGL(tsvc_2_s152_fp64_kernel, dim3(blocks), dim3(threadsPerBlock), 0, 0,
                       a, b, c, d, e, LEN_1D);
    // Ensure kernel completion before returning
    hipDeviceSynchronize();
}
