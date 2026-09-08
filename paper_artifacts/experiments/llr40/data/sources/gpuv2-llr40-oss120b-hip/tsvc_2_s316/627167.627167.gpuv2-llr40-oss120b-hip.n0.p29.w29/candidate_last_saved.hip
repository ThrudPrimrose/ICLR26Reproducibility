#include <hip/hip_runtime.h>
#include <cstdint>
#include <cfloat>
#include <vector>
#include <algorithm>

// CUDA/HIP kernel for block-wise minimum reduction on a double array.
// Each block computes the minimum of a chunk of the array using a grid-stride loop.
// The per-block minima are written to blockMins[blockIdx.x].

extern "C" __global__ void min_reduce_kernel(const double* __restrict__ a,
                                            double* __restrict__ blockMins,
                                            int64_t N) {
    // Use a fixed block size of 256 threads.
    // Shared memory for reduction.
    __shared__ double sdata[256]; // blockDim.x must be <= 256.

    // Global thread index.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Initialize local minimum to a very large value.
    double local_min = DBL_MAX;

    // Grid-stride loop over the input array.
    for (int64_t i = idx; i < N; i += gridDim.x * blockDim.x) {
        double val = a[i];
        if (val < local_min) {
            local_min = val;
        }
    }

    // Store the local minimum in shared memory.
    sdata[threadIdx.x] = local_min;
    __syncthreads();

    // Parallel reduction in shared memory.
    // Reduce within the block using binary tree.
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            double other = sdata[threadIdx.x + stride];
            if (other < sdata[threadIdx.x]) {
                sdata[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }

    // Write the result for this block to global memory.
    if (threadIdx.x == 0) {
        blockMins[blockIdx.x] = sdata[0];
    }
}

// Host wrapper function matching the expected C ABI.
extern "C" void tsvc_2_s316_fp64(const double* __restrict__ a,
                                 double* __restrict__ result,
                                 const int64_t LEN_1D) {
    // Edge case: if the array has length zero, return +inf.
    if (LEN_1D <= 0) {
        result[0] = DBL_MAX;
        return;
    }

    // Choose a block size. 256 is a good default for most GPUs.
    const int blockSize = 256;
    // Compute number of blocks needed to cover the array.
    int64_t numBlocks = (LEN_1D + blockSize - 1) / blockSize;
    // Limit the number of blocks to a reasonable maximum (e.g., 65535)
    // to stay within launch limits on older devices.
    const int maxBlocks = 65535;
    if (numBlocks > maxBlocks) {
        numBlocks = maxBlocks;
    }

    // Allocate device buffer for per-block minima.
    double* d_blockMins = nullptr;
    hipError_t err = hipMalloc(&d_blockMins, numBlocks * sizeof(double));
    if (err != hipSuccess) {
        // Allocation failed – signal error by writing max value.
        result[0] = DBL_MAX;
        return;
    }

    // Launch the reduction kernel.
    dim3 grid((unsigned int)numBlocks);
    dim3 block(blockSize);
    // No dynamic shared memory needed because we use static __shared__ array.
    hipLaunchKernelGGL(min_reduce_kernel,
                       grid,
                       block,
                       0, // shared memory (bytes)
                       0, // stream
                       a,
                       d_blockMins,
                       LEN_1D);
    // Wait for kernel completion.
    err = hipGetLastError();
    if (err != hipSuccess) {
        hipFree(d_blockMins);
        result[0] = DBL_MAX;
        return;
    }
    hipDeviceSynchronize();

    // Copy per-block minima back to host.
    std::vector<double> h_blockMins((size_t)numBlocks);
    err = hipMemcpy(h_blockMins.data(), d_blockMins, numBlocks * sizeof(double), hipMemcpyDeviceToHost);
    hipFree(d_blockMins);
    if (err != hipSuccess) {
        result[0] = DBL_MAX;
        return;
    }

    // Reduce on the host to obtain the final minimum.
    double min_val = h_blockMins[0];
    for (int64_t i = 1; i < numBlocks; ++i) {
        if (h_blockMins[i] < min_val) {
            min_val = h_blockMins[i];
        }
    }
    result[0] = min_val;
}

