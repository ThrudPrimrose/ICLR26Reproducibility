#include <hip/hip_runtime.h>
#include <cstdint>
#include <limits>

// Structure to hold a candidate max value and its flat index.
struct MaxIdx {
    double val;
    int64_t idx;
};

// Comparator that chooses the larger value; for equal values picks the smaller index
__device__ __forceinline__ MaxIdx maxIdxCombine(const MaxIdx &a, const MaxIdx &b) {
    if (a.val > b.val) return a;
    if (b.val > a.val) return b;
    // values equal, pick the smaller index (earlier in row-major order)
    return (a.idx <= b.idx) ? a : b;
}

// First kernel: reduce directly from the input array to per-block results.
// Each thread processes a strided chunk of the input.
__global__ void reduce_input(const double * __restrict__ aa, MaxIdx * __restrict__ out,
                             int64_t LEN_2D, int64_t N) {
    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    // Global linear thread index
    int64_t gid = static_cast<int64_t>(blockIdx.x) * blockSize + tid;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockSize;

    // Initialise per-thread local maximum
    MaxIdx local;
    local.val = -std::numeric_limits<double>::infinity();
    local.idx = -1;

    for (int64_t idx = gid; idx < N; idx += stride) {
        double v = aa[idx];
        if (v > local.val) {
            local.val = v;
            local.idx = idx;
        } else if (v == local.val && idx < local.idx) {
            // Tie-breaking: keep the smaller (earlier) index
            local.idx = idx;
        }
    }

    // Shared memory reduction within the block
    __shared__ MaxIdx sdata[256]; // assuming blockDim.x <= 256
    sdata[tid] = local;
    __syncthreads();

    // Reduce in shared memory (binary tree)
    for (int offset = blockSize / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] = maxIdxCombine(sdata[tid], sdata[tid + offset]);
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

// Generic reduction kernel for MaxIdx arrays.
__global__ void reduce_maxidx(const MaxIdx * __restrict__ in, MaxIdx * __restrict__ out,
                              int64_t N) {
    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    int64_t gid = static_cast<int64_t>(blockIdx.x) * blockSize + tid;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockSize;

    MaxIdx local;
    local.val = -std::numeric_limits<double>::infinity();
    local.idx = -1;

    for (int64_t i = gid; i < N; i += stride) {
        MaxIdx cand = in[i];
        if (cand.val > local.val) {
            local = cand;
        } else if (cand.val == local.val && cand.idx < local.idx) {
            local = cand;
        }
    }

    __shared__ MaxIdx sdata[256]; // blockDim.x <= 256
    sdata[tid] = local;
    __syncthreads();

    for (int offset = blockSize / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] = maxIdxCombine(sdata[tid], sdata[tid + offset]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

// Final kernel: compute checksum and store into output array.
__global__ void write_result(const MaxIdx * __restrict__ best, double * __restrict__ bb,
                             int64_t LEN_2D) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        MaxIdx b = best[0];
        int64_t i = b.idx / LEN_2D;
        int64_t j = b.idx % LEN_2D;
        double chksum = b.val + static_cast<double>(i) + static_cast<double>(j);
        bb[0] = chksum;
    }
}

extern "C" void tsvc_2_s3110_fp64(const double * __restrict__ aa,
                                   double * __restrict__ bb,
                                   const int64_t LEN_2D) {
    // Total number of elements in the 2D array.
    int64_t N = LEN_2D * LEN_2D;
    // Choose a block size; 256 threads works well for most GPUs.
    const int BLOCK_SIZE = 256;
    int64_t gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Cap gridSize to avoid launching more blocks than needed for tiny inputs.
    if (gridSize < 1) gridSize = 1;

    // Allocate temporary storage for the reduction.
    MaxIdx *d_buf0;
    MaxIdx *d_buf1;
    // Allocate enough space for the maximum possible number of blocks.
    hipMalloc(&d_buf0, gridSize * sizeof(MaxIdx));
    hipMalloc(&d_buf1, gridSize * sizeof(MaxIdx));

    // First reduction from raw data.
    reduce_input<<<gridSize, BLOCK_SIZE>>>(aa, d_buf0, LEN_2D, N);
    hipGetLastError(); // launch check
    // Iteratively reduce until we have a single element.
    int64_t cur_n = gridSize;
    MaxIdx *cur_in = d_buf0;
    MaxIdx *cur_out = d_buf1;
    while (cur_n > 1) {
        int64_t next_grid = (cur_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        reduce_maxidx<<<next_grid, BLOCK_SIZE>>>(cur_in, cur_out, cur_n);
        hipGetLastError();
        // swap buffers
        MaxIdx *tmp = cur_in;
        cur_in = cur_out;
        cur_out = tmp;
        cur_n = next_grid;
    }

    // Compute checksum and write to output.
    write_result<<<1, 1>>>(cur_in, bb, LEN_2D);
    hipGetLastError();

    // Free temporary buffers.
    hipFree(d_buf0);
    hipFree(d_buf1);
}

