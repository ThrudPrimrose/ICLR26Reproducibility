#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(call)                                                       \
    do {                                                                      \
        hipError_t err = call;                                                \
        if (err != hipSuccess) {                                             \
            fprintf(stderr, "HIP error: %s:%d: %s\n", __FILE__, __LINE__,     \
                    hipGetErrorString(err));                                 \
            exit(1);                                                         \
        }                                                                     \
    } while (0)

// Device kernel implementing the reference algorithm.
__global__ void tsvc_2_s2710_kernel(double* __restrict__ a,
                                    double* __restrict__ b,
                                    double* __restrict__ c,
                                    const double* __restrict__ d,
                                    const double* __restrict__ e,
                                    const double* __restrict__ x,
                                    int64_t LEN_1D) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < LEN_1D; i += stride) {
        if (a[i] > b[i]) {
            a[i] += b[i] * d[i];
            if (LEN_1D > 10) {
                c[i] += d[i] * d[i];
            } else {
                c[i] = d[i] * e[i] + 1.0;
            }
        } else {
            b[i] = a[i] + e[i] * e[i];
            if (x[0] > 0.0) {
                c[i] = a[i] + d[i] * d[i];
            } else {
                c[i] += e[i] * e[i];
            }
        }
    }
}

extern "C" void tsvc_2_s2710_fp64(double* a,
                                 double* b,
                                 double* c,
                                 const double* d,
                                 const double* e,
                                 const double* x,
                                 int64_t LEN_1D) {
    const int blockSize = 256;
    // Compute grid size; ensure at least one block.
    int gridSize = static_cast<int>((LEN_1D + blockSize - 1) / blockSize);
    if (gridSize < 1) gridSize = 1;
    // Launch the kernel.
    hipLaunchKernelGGL(tsvc_2_s2710_kernel,
                       dim3(gridSize), dim3(blockSize), 0, 0,
                       a, b, c, d, e, x, LEN_1D);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
}

