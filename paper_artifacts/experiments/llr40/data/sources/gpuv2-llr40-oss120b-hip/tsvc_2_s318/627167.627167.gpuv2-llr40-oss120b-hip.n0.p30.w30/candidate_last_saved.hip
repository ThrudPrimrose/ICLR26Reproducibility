#include <hip/hip_runtime.h>
#include <cstdint>
#include <cmath>
#include <cstdlib>

// Kernel to compute per-block maximum absolute value and its index.
// a       : input array (device pointer)
// block_max: per-block maximum absolute value (output, device)
// block_idx: per-block index of that maximum (output, device)
// LEN     : number of logical elements (LEN_1D)
// inc     : stride between elements in the input array
__global__ void tsvc_2_s318_reduce_kernel(const double* __restrict__ a,
                                           double* __restrict__ block_max,
                                           int64_t* __restrict__ block_idx,
                                           int64_t LEN,
                                           int64_t inc) {
    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int64_t stride = static_cast<int64_t>(blockSize) * gridDim.x;
    const int64_t globalId = static_cast<int64_t>(blockIdx.x) * blockSize + tid;

    double local_max = -INFINITY; // start with the smallest possible value
    int64_t local_idx = -1;

    // Loop over the logical indices guarded by the total stride.
    for (int64_t i = globalId; i < LEN; i += stride) {
        double v = fabs(a[i * inc]);
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
        // If v == local_max we keep the earlier (smaller) index – nothing to do.
    }

    // Shared memory reduction within the block.
    extern __shared__ char shared_mem[]; // allocate dynamically
    double* s_max = reinterpret_cast<double*>(shared_mem);
    int64_t* s_idx = reinterpret_cast<int64_t*>(s_max + blockSize);

    s_max[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    // Reduce with power-of-two steps.
    for (int step = blockSize / 2; step > 0; step >>= 1) {
        if (tid < step) {
            double cand_max = s_max[tid + step];
            int64_t cand_idx = s_idx[tid + step];
            double cur_max = s_max[tid];
            int64_t cur_idx = s_idx[tid];
            if (cand_max > cur_max) {
                s_max[tid] = cand_max;
                s_idx[tid] = cand_idx;
            } else if (cand_max == cur_max && cand_idx < cur_idx) {
                // Tie: choose smaller index (earlier occurrence)
                s_idx[tid] = cand_idx;
            }
        }
        __syncthreads();
    }

    // Write the block's result.
    if (tid == 0) {
        block_max[blockIdx.x] = s_max[0];
        block_idx[blockIdx.x] = s_idx[0];
    }
}

extern "C" __attribute__((visibility("default")))
void tsvc_2_s318_fp64(const double* __restrict__ a,
                       double* __restrict__ result,
                       const int64_t LEN_1D,
                       const int64_t inc) {
    // Edge case: empty input
    if (LEN_1D <= 0) {
        if (result) result[0] = 0.0;
        return;
    }

    const int threadsPerBlock = 256;
    // Determine number of blocks; cap to a reasonable maximum (e.g., 65535).
    int64_t blocks = (LEN_1D + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks > 65535) blocks = 65535;

    // Allocate space for per‑block results.
    double* d_block_max = nullptr;
    int64_t* d_block_idx = nullptr;
    hipMalloc(&d_block_max, blocks * sizeof(double));
    hipMalloc(&d_block_idx, blocks * sizeof(int64_t));

    // Shared memory size: double[threadsPerBlock] + int64_t[threadsPerBlock]
    size_t sharedBytes = threadsPerBlock * (sizeof(double) + sizeof(int64_t));
    hipLaunchKernelGGL(tsvc_2_s318_reduce_kernel,
                       dim3(blocks), dim3(threadsPerBlock),
                       sharedBytes, 0,
                       a, d_block_max, d_block_idx, LEN_1D, inc);
    hipDeviceSynchronize();

    // Copy per‑block results back to host.
    double* h_block_max = (double*)malloc(blocks * sizeof(double));
    int64_t* h_block_idx = (int64_t*)malloc(blocks * sizeof(int64_t));
    hipMemcpy(h_block_max, d_block_max, blocks * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(h_block_idx, d_block_idx, blocks * sizeof(int64_t), hipMemcpyDeviceToHost);

    // Free device buffers.
    hipFree(d_block_max);
    hipFree(d_block_idx);

    // Reduce on host to obtain the global maximum and its index.
    double maxv = -INFINITY;
    int64_t max_idx = -1;
    for (int64_t b = 0; b < blocks; ++b) {
        double v = h_block_max[b];
        int64_t idx = h_block_idx[b];
        if (v > maxv) {
            maxv = v;
            max_idx = idx;
        } else if (v == maxv && idx < max_idx) {
            max_idx = idx;
        }
    }

    // Free host buffers.
    free(h_block_max);
    free(h_block_idx);

    // According to the reference, result = maxv + (double)index.
    double out = maxv + static_cast<double>(max_idx);
    result[0] = out;
}

