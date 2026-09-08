#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Forward declaration of the kernel
__global__ void update_kernel(double* __restrict__ a,
                              const double* __restrict__ aa,
                              const int64_t LEN_2D,
                              const int64_t j);

extern "C" void tsvc_2_s115_fp64(double* __restrict__ a,
                                 const double* __restrict__ aa,
                                 const int64_t LEN_2D) {
    for (int64_t j = 0; j < LEN_2D; ++j) {
        int64_t n = LEN_2D - (j + 1);
        if (n <= 0) continue;
        const int blockSize = 256;
        int64_t gridSize = (n + blockSize - 1) / blockSize;
        hipLaunchKernelGGL(update_kernel,
                           dim3(gridSize), dim3(blockSize), 0, 0,
                           a, aa, LEN_2D, j);
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            fprintf(stderr, "HIP kernel launch error: %s\n", hipGetErrorString(err));
            std::abort();
        }
        hipDeviceSynchronize();
    }
}

__global__ void update_kernel(double* __restrict__ a,
                               const double* __restrict__ aa,
                               const int64_t LEN_2D,
                               const int64_t j) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t i = j + 1 + idx;
    if (i < LEN_2D) {
        double a_j = a[j]; // read a[j]
        double val = aa[j * LEN_2D + i] * a_j;
        a[i] -= val;
    }
}
