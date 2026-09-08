#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdint.h>

// Kernel to perform a[i] = a[i+1] + b[i] for i = 0..LEN_1D-2
__global__ void ext_war_unit_kernel(double* __restrict__ a, const double* __restrict__ b, int64_t len) {
    int64_t blockStart = (int64_t)blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int64_t i = blockStart + tid;
    // Use dynamic shared memory of size blockDim.x + 1 doubles
    extern __shared__ double sh[]; // sh[0..blockDim.x]
    // Load a[i] into shared memory if within range
    if (i < len) {
        sh[tid] = a[i];
    }
    // Load extra element (a[blockStart + blockDim.x] or last element) into shared memory slot blockDim.x
    if (tid == blockDim.x - 1) {
        int64_t extraIdx = blockStart + blockDim.x;
        if (extraIdx >= len) extraIdx = len - 1; // clamp to last valid index
        sh[tid + 1] = a[extraIdx];
    }
    // Ensure all loads are visible
    __syncthreads();
    // Process only valid indices (i < len-1)
    if (i >= len - 1) return;
    double a_ip1 = sh[tid + 1];
    double result = a_ip1 + b[i];
    a[i] = result;
}

extern "C" void ext_war_unit_fp64(double* __restrict__ a, const double* __restrict__ b, int64_t LEN_1D) {
    // No work if LEN_1D <= 1
    if (LEN_1D <= 1) return;
    const int threads_per_block = 256;
    int64_t N = LEN_1D - 1; // number of iterations
    int blocks = (int)((N + threads_per_block - 1) / threads_per_block);
    // Launch kernel with dynamic shared memory of (threads_per_block + 1) doubles
    size_t shmem_bytes = (threads_per_block + 1) * sizeof(double);
    hipLaunchKernelGGL(ext_war_unit_kernel,
                       dim3(blocks), dim3(threads_per_block), shmem_bytes, 0,
                       a, b, LEN_1D);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "HIP kernel launch error: %s\n", hipGetErrorString(err));
        exit(1);
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "HIP device sync error: %s\n", hipGetErrorString(err));
        exit(1);
    }
}

