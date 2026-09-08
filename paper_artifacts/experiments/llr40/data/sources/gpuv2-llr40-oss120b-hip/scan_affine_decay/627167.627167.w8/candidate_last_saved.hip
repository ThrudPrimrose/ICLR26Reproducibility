#include <hip/hip_runtime.h>
#include <cstdint>

// Structure representing the affine transformation (a * y + b)
struct Pair {
    double a; // multiplicative coefficient
    double b; // additive coefficient
};

// Combine two transformations: apply 'earlier' then 'later'
__device__ __host__ inline Pair combine(const Pair &later, const Pair &earlier) {
    Pair res;
    res.a = later.a * earlier.a;
    res.b = later.a * earlier.b + later.b;
    return res;
}

// First kernel: compute per-element prefix transformations and block totals
extern "C" __global__ void scan_prefix_kernel(const double* __restrict__ c,
                                                const double* __restrict__ x,
                                                Pair* __restrict__ out_prefix,
                                                Pair* __restrict__ block_sum,
                                                int64_t N) {
    // Shared memory for the block
    extern __shared__ Pair sh[]; // size = blockDim.x
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load the local transformation (identity for out‑of‑range threads)
    Pair my;
    if (idx < N) {
        my.a = c[idx];
        my.b = x[idx];
    } else {
        my.a = 1.0; // identity multiplicative
        my.b = 0.0; // identity additive
    }
    sh[tid] = my;
    __syncthreads();

    // Hillis‑Steele inclusive scan (non‑commutative but associative)
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        Pair val = sh[tid];
        if (tid >= offset) {
            // apply the transformation of the earlier element first
            val = combine(sh[tid], sh[tid - offset]);
        }
        __syncthreads();
        if (tid >= offset) {
            sh[tid] = val;
        }
        __syncthreads();
    }

    // Write the per‑element prefix (still in device order)
    if (idx < N) {
        out_prefix[idx] = sh[tid];
    }

    // Compute block‑level total transformation (the prefix of the last valid element)
    if (tid == 0) {
        int block_start = blockIdx.x * blockDim.x;
        int block_end = min<int64_t>(N, block_start + blockDim.x); // exclusive
        Pair block_res;
        if (block_end > block_start) {
            // The last element's prefix is stored at position (block_end - block_start - 1)
            block_res = sh[block_end - block_start - 1];
        } else {
            // Empty block -> identity
            block_res.a = 1.0;
            block_res.b = 0.0;
        }
        block_sum[blockIdx.x] = block_res;
    }
}

// Second kernel: combine block prefixes with per‑element prefixes to produce final y
extern "C" __global__ void final_y_kernel(double* __restrict__ y,
                                           const Pair* __restrict__ prefix,
                                           const Pair* __restrict__ block_prefix,
                                           double y0,
                                           int64_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        Pair p_local = prefix[idx];
        Pair p_block = block_prefix[blockIdx.x];
        // total transformation for this element = combine(p_local, p_block)
        Pair total;
        total.a = p_local.a * p_block.a;
        total.b = p_local.a * p_block.b + p_local.b;
        // write result for position idx+1 (because index 0 is the seed)
        y[idx + 1] = total.a * y0 + total.b;
    }
}

// Exported entry point matching the C‑ABI expected by the benchmark harness
extern "C" void scan_affine_decay(double* __restrict__ y,
                                   const double* __restrict__ c,
                                   const double* __restrict__ x,
                                   const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;
    // Copy the first element (seed) directly on the device
    hipMemcpy(y, x, sizeof(double), hipMemcpyDeviceToDevice);
    if (LEN_1D == 1) return;

    const int64_t N = LEN_1D - 1; // number of transformations to process
    const int BLOCK = 256;
    int blocks = static_cast<int>((N + BLOCK - 1) / BLOCK);

    // Allocate temporary buffers on the device
    Pair* d_prefix;          // per‑element prefix transformations
    Pair* d_block_sum;       // block‑level total transformations
    hipMalloc(&d_prefix, N * sizeof(Pair));
    hipMalloc(&d_block_sum, blocks * sizeof(Pair));

    // Kernel 1 – compute per‑element prefixes and block totals
    hipLaunchKernelGGL(scan_prefix_kernel,
                       dim3(blocks), dim3(BLOCK), BLOCK * sizeof(Pair), 0,
                       c + 1, x + 1, d_prefix, d_block_sum, N);
    hipGetLastError();
    hipDeviceSynchronize();

    // Transfer block totals to the host for a cheap sequential prefix
    Pair* h_block_sum = (Pair*)malloc(blocks * sizeof(Pair));
    hipMemcpy(h_block_sum, d_block_sum, blocks * sizeof(Pair), hipMemcpyDeviceToHost);

    // Compute exclusive prefix of block totals on the host (identity for first block)
    Pair* h_block_prefix = (Pair*)malloc(blocks * sizeof(Pair));
    Pair accum; // identity transformation
    accum.a = 1.0;
    accum.b = 0.0;
    for (int i = 0; i < blocks; ++i) {
        h_block_prefix[i] = accum; // prefix before block i
        // accum = combine(block_sum[i], accum)
        Pair bs = h_block_sum[i];
        Pair next;
        next.a = bs.a * accum.a;
        next.b = bs.a * accum.b + bs.b;
        accum = next;
    }

    // Upload block prefixes to the device
    Pair* d_block_prefix;
    hipMalloc(&d_block_prefix, blocks * sizeof(Pair));
    hipMemcpy(d_block_prefix, h_block_prefix, blocks * sizeof(Pair), hipMemcpyHostToDevice);

    // Retrieve the seed value y0 from device memory (single double copy)
    double y0;
    hipMemcpy(&y0, x, sizeof(double), hipMemcpyDeviceToHost);

    // Kernel 2 – compute final y values using the combined transformations
    hipLaunchKernelGGL(final_y_kernel,
                       dim3(blocks), dim3(BLOCK), 0, 0,
                       y, d_prefix, d_block_prefix, y0, N);
    hipGetLastError();
    hipDeviceSynchronize();

    // Clean up temporary allocations
    hipFree(d_prefix);
    hipFree(d_block_sum);
    hipFree(d_block_prefix);
    free(h_block_sum);
    free(h_block_prefix);
}

