#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>

#define HIP_CHECK(call)                                   \
    do {                                                  \
        hipError_t err = call;                            \
        if (err != hipSuccess) {                          \
            fprintf(stderr, "HIP error at %s:%d: %s\n",   \
                    __FILE__, __LINE__, hipGetErrorString(err)); \
            exit(1);                                     \
        }                                                 \
    } while (0)

// Device kernel performing elementwise operations and per-thread partial sum.
__global__ void tsvc_2_s319_kernel(double* __restrict__ a,
                                            double* __restrict__ b,
                                            const double* __restrict__ c,
                                            const double* __restrict__ d,
                                            const double* __restrict__ e,
                                            int64_t LEN_1D,
                                            double* __restrict__ block_sums) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    double local_sum = 0.0;
    for (int64_t i = idx; i < LEN_1D; i += stride) {
        a[i] = c[i] + d[i];
        local_sum += a[i];
        b[i] = c[i] + e[i];
        local_sum += b[i];
    }
    // Reduce within block using shared memory.
    __shared__ double shmem[256]; // assume max 256 threads per block
    shmem[threadIdx.x] = local_sum;
    __syncthreads();
    // Tree reduction.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shmem[threadIdx.x] += shmem[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = shmem[0];
    }
}

extern "C" void tsvc_2_s319_fp64(double* __restrict__ a,
                                 double* __restrict__ b,
                                 const double* __restrict__ c,
                                 const double* __restrict__ d,
                                 const double* __restrict__ e,
                                 const int64_t LEN_1D) {
    // Determine launch configuration.
    const int threads_per_block = 256;
    int64_t blocks = (LEN_1D + threads_per_block - 1) / threads_per_block;
    if (blocks < 1) blocks = 1;
    // Allocate temporary buffer for block sums.
    double* d_block_sums = nullptr;
    HIP_CHECK(hipMalloc(&d_block_sums, blocks * sizeof(double)));
    // Launch kernel.
    dim3 grid(blocks);
    dim3 block(threads_per_block);
    hipLaunchKernelGGL(tsvc_2_s319_kernel,
                       grid, block, 0, 0,
                       a, b, c, d, e, LEN_1D, d_block_sums);
    // Ensure kernel completion.
    HIP_CHECK(hipDeviceSynchronize());
    // Copy block sums back to host.
    std::vector<double> h_block_sums(blocks);
    HIP_CHECK(hipMemcpy(h_block_sums.data(), d_block_sums, blocks * sizeof(double), hipMemcpyDeviceToHost));
    // Free device buffer for block sums.
    HIP_CHECK(hipFree(d_block_sums));
    // Reduce on host.
    double total = 0.0;
    for (int64_t i = 0; i < blocks; ++i) {
        total += h_block_sums[i];
    }
    // Copy total back to b[0] on device.
    HIP_CHECK(hipMemcpy(b, &total, sizeof(double), hipMemcpyHostToDevice));
}

