#include <hip/hip_runtime.h>
#include <cstdint>
#include <cmath>

// Device kernel: computes per-block max and index
__global__ void argmax_with_index_kernel(const double *a, int64_t *out_idx, double *out_val, int64_t len) {
    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    double local_max = -INFINITY;
    int64_t local_idx = -1;
    // Stride over the array
    for (int64_t i = tid; i < len; i += blockSize) {
        double val = a[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }
    // Shared memory for reduction
    __shared__ double s_val[256]; // assume max block size 256
    __shared__ int64_t s_idx[256];
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();
    // Reduction to find max within block
    for (int step = blockSize / 2; step > 0; step >>= 1) {
        if (tid < step) {
            double other_val = s_val[tid + step];
            int64_t other_idx = s_idx[tid + step];
            if (other_val > s_val[tid]) {
                s_val[tid] = other_val;
                s_idx[tid] = other_idx;
            }
        }
        __syncthreads();
    }
    // Write result from thread 0
    if (tid == 0) {
        *out_val = s_val[0];
        *out_idx = s_idx[0];
    }
}

extern "C" __attribute__((visibility("default")))
void argmax_with_index_fp64(const double *a, int64_t *out_index, double *out_value, const int64_t LEN_1D) {
    // Launch with a single block; choose 256 threads or less if LEN_1D is smaller.
    int threads = 256;
    if (LEN_1D < threads) {
        threads = static_cast<int>(LEN_1D);
        if (threads == 0) threads = 1; // avoid zero threads
    }
    dim3 blockDim(threads);
    dim3 gridDim(1);
    hipLaunchKernelGGL(argmax_with_index_kernel, gridDim, blockDim, 0, 0, a, out_index, out_value, LEN_1D);
    // Ensure kernel completion before returning.
    hipDeviceSynchronize();
}
