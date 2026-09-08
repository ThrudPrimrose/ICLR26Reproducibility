#include <hip/hip_runtime.h>
#include <cstdint>

// Device kernel to compute partial sum of positive elements
__global__ void tsvc_2_s3111_kernel(const double * __restrict__ a, double * __restrict__ b, int64_t LEN_1D) {
    extern __shared__ double sdata[]; // shared memory size = blockDim.x * sizeof(double)
    int tid = threadIdx.x;
    int64_t i = blockIdx.x * blockDim.x + tid;
    double val = 0.0;
    if (i < LEN_1D) {
        double ai = a[i];
        if (ai > 0.0) {
            val = ai;
        }
    }
    sdata[tid] = val;
    __syncthreads();
    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        // Atomically add block's partial sum to global result
        atomicAdd(b, sdata[0]);
    }
}

extern "C" void tsvc_2_s3111_fp64(const double * __restrict__ a, double * __restrict__ b, const int64_t LEN_1D) {
    // Zero output accumulator
    b[0] = 0.0;
    // Choose block size - power of two for reduction
    int blockSize = 256;
    int gridSize = (LEN_1D + blockSize - 1) / blockSize;
    // Launch kernel with shared memory equal to blockSize * sizeof(double)
    hipLaunchKernelGGL(tsvc_2_s3111_kernel, dim3(gridSize), dim3(blockSize), blockSize * sizeof(double), 0,
                       a, b, LEN_1D);
    // Wait for kernel completion to ensure b[0] is ready when returning
    hipDeviceSynchronize();
}
