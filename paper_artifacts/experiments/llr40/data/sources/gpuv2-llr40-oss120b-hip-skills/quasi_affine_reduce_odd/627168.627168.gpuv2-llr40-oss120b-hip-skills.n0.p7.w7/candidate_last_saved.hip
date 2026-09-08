#include <hip/hip_runtime.h>
#include <cstdio>

#define HIP_CHECK(call) do { hipError_t err = call; if (err != hipSuccess) { printf("HIP error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); } } while(0)

extern "C" __global__ void reduce_odd_kernel(const double* __restrict__ a, double* __restrict__ out, long long LEN_1D) {
    // Shared memory for block reduction
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;
    // Compute global odd index for this thread
    long long idx = 1 + 2LL * ((long long)blockIdx.x * blockSize + tid);
    double val = 0.0;
    if (idx < LEN_1D) {
        val = a[idx];
    }
    sdata[tid] = val;
    __syncthreads();
    // Reduction in shared memory
    for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        // Atomic add block sum to out[0]
        atomicAdd(out, sdata[0]);
    }
}

extern "C" void quasi_affine_reduce_odd_fp64(const double* __restrict__ a, double* __restrict__ out, const long long LEN_1D) {
    // Handle case with no odd elements
    if (LEN_1D <= 1) {
        HIP_CHECK(hipMemset(out, 0, sizeof(double)));
        return;
    }
    // Initialize output to zero
    HIP_CHECK(hipMemset(out, 0, sizeof(double)));
    const unsigned int blockSize = 256;
    long long oddCount = LEN_1D / 2; // number of odd indices
    long long numBlocks = (oddCount + blockSize - 1) / blockSize;
    dim3 grid((unsigned int)numBlocks);
    dim3 block(blockSize);
    // Launch kernel with dynamic shared memory
    hipLaunchKernelGGL(reduce_odd_kernel, grid, block, blockSize * sizeof(double), 0,
                       a, out, LEN_1D);
    // Check for launch errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        printf("HIP launch error: %s\n", hipGetErrorString(err));
    }
    // Wait for kernel to finish
    HIP_CHECK(hipDeviceSynchronize());
}
