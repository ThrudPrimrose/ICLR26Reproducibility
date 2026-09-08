#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" __global__ void sum_kernel_fp64(const double* __restrict__ a, double* __restrict__ partial_sums, int64_t LEN) {
    double sum = 0.0;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = tid; i < LEN; i += stride) {
        double v = a[i];
        if (v > 0.0) sum += v;
    }
    // reduction within block using shared memory
    extern __shared__ double sdata[]; // dynamic shared memory
    int lane = threadIdx.x;
    sdata[lane] = sum;
    __syncthreads();
    // binary tree reduction
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            sdata[lane] += sdata[lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

extern "C" void tsvc_2_s3111_fp64(const double* __restrict__ a, double* __restrict__ b, const int64_t LEN_1D) {
    const int blockSize = 256;
    int gridSize = (LEN_1D + blockSize - 1) / blockSize;
    // allocate buffer for block partial sums
    double* d_partial;
    hipError_t err;
    err = hipMalloc(&d_partial, gridSize * sizeof(double));
    if (err != hipSuccess) {
        fprintf(stderr, "hipMalloc failed: %s\n", hipGetErrorString(err));
        exit(1);
    }
    // launch kernel
    hipLaunchKernelGGL(sum_kernel_fp64, dim3(gridSize), dim3(blockSize), blockSize * sizeof(double), 0, a, d_partial, LEN_1D);
    // copy back and accumulate on host
    double* h_partial = (double*)malloc(gridSize * sizeof(double));
    err = hipMemcpy(h_partial, d_partial, gridSize * sizeof(double), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        fprintf(stderr, "hipMemcpy (D2H) failed: %s\n", hipGetErrorString(err));
        exit(1);
    }
    double total = 0.0;
    for (int i = 0; i < gridSize; ++i) {
        total += h_partial[i];
    }
    b[0] = total;
    free(h_partial);
    err = hipFree(d_partial);
    if (err != hipSuccess) {
        fprintf(stderr, "hipFree failed: %s\n", hipGetErrorString(err));
        exit(1);
    }
}
