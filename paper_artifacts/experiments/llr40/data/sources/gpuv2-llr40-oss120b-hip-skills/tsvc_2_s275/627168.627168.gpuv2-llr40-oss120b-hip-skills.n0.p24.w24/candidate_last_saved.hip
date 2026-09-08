#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdint.h>

extern "C" __global__ void tsvc_2_s275_fp64_kernel(double* __restrict__ aa, const double* __restrict__ bb, const double* __restrict__ cc, const int64_t LEN_2D) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= LEN_2D) return;
    double a0 = aa[i];
    if (a0 > 0.0) {
        double prev = a0;
        for (int64_t j = 1; j < LEN_2D; ++j) {
            double b = bb[j * LEN_2D + i];
            double c = cc[j * LEN_2D + i];
            double val = prev + b * c;
            aa[j * LEN_2D + i] = val;
            prev = val;
        }
    }
}

extern "C" void tsvc_2_s275_fp64(double* __restrict__ aa, const double* __restrict__ bb, const double* __restrict__ cc, const int64_t LEN_2D) {
    const int threads = 256;
    const int blocks = (int)((LEN_2D + threads - 1) / threads);
    hipLaunchKernelGGL(tsvc_2_s275_fp64_kernel, dim3(blocks), dim3(threads), 0, 0, aa, bb, cc, LEN_2D);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "hipLaunchKernelGGL error: %s\n", hipGetErrorString(err));
        exit(1);
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "hipDeviceSynchronize error: %s\n", hipGetErrorString(err));
        exit(1);
    }
}
