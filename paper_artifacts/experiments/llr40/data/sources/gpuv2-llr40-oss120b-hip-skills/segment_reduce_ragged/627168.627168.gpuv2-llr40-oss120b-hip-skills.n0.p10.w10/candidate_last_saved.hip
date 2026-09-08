#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

// Macro for error checking HIP API calls.
#define HIP_CHECK(call)                                                      \
    do {                                                                     \
        hipError_t err = call;                                               \
        if (err != hipSuccess) {                                            \
            fprintf(stderr, "HIP error: %s:%d: %s\n", __FILE__, __LINE__,   \
                    hipGetErrorString(err));                               \
            exit(1);                                                       \
        }                                                                    \
    } while (0)

// Device kernel that computes a segmented dot product (val * w) per segment.
// Each block processes one segment (identified by blockIdx.x).
// The reduction is performed in shared memory; the final result for each segment
// is written to out[segment].
extern "C" __global__ void segment_reduce_ragged_kernel(const int64_t* __restrict__ row_ptr,
                                                         const double* __restrict__ val,
                                                         const double* __restrict__ w,
                                                         double* __restrict__ out) {
    // Identify segment.
    const int64_t seg = static_cast<int64_t>(blockIdx.x);
    const int64_t start = row_ptr[seg];
    const int64_t end   = row_ptr[seg + 1];

    double thread_sum = 0.0;
    // Each thread walks over its portion of the segment with a stride of blockDim.x.
    for (int64_t i = start + threadIdx.x; i < end; i += blockDim.x) {
        thread_sum += val[i] * w[i];
    }

    // Parallel reduction within the block using shared memory.
    extern __shared__ double shmem[];
    shmem[threadIdx.x] = thread_sum;
    __syncthreads();

    // Reduce in-place; blockDim.x is assumed to be a power of two (e.g., 256).
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shmem[threadIdx.x] += shmem[threadIdx.x + offset];
        }
        __syncthreads();
    }

    // Write the result for the segment.
    if (threadIdx.x == 0) {
        out[seg] = shmem[0];
    }
}

// Host entry point matching the reference signature.
extern "C" void segment_reduce_ragged(const int64_t* __restrict__ row_ptr,
                                       const double* __restrict__ val,
                                       const double* __restrict__ w,
                                       double* __restrict__ out,
                                       const int64_t NSEG) {
    // Determine launch configuration: one block per segment.
    const int threads_per_block = 256;
    int blocks = static_cast<int>(NSEG);
    if (blocks < 1) blocks = 1;

    // Launch kernel with dynamic shared memory equal to threads_per_block * sizeof(double).
    dim3 grid_dim(static_cast<unsigned int>(blocks));
    dim3 block_dim(static_cast<unsigned int>(threads_per_block));
    size_t shmem_bytes = threads_per_block * sizeof(double);

    hipLaunchKernelGGL(segment_reduce_ragged_kernel,
                       grid_dim, block_dim, shmem_bytes, 0,
                       row_ptr, val, w, out);
    // Check for launch errors and synchronize.
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
}

