#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Optimized single-block kernel that processes the triangular wavefront.
// It stores the previous anti‑diagonal in shared memory, so each element
// reads only its own original value from global memory and the two required
// dependencies from the shared buffer.
extern "C" __global__ void wf_triangular_kernel(double * __restrict__ a, int64_t L) {
    // One block is sufficient; we use __syncthreads() for intra‑block sync.
    // The block size can be up to the hardware limit (1024). Choose a reasonable size.
    const int64_t tid = static_cast<int64_t>(threadIdx.x);
    const int64_t blockSize = static_cast<int64_t>(blockDim.x);

    // Allocate two double buffers in shared memory for the previous and current diagonals.
    // The maximum diagonal length is ceil(L/2). We allocate space for both buffers.
    extern __shared__ double shared_buf[];
    // Compute the maximum possible diagonal length (ceil(L/2)).
    const int64_t max_diag = (L + 1) / 2 + 1; // +1 for safety.
    double *prev = shared_buf;                       // size = max_diag
    double *cur  = shared_buf + max_diag;            // size = max_diag

    // No previous diagonal for s = 2, so we will read dependencies directly from global memory.
    // Iterate over anti‑diagonals (wavefronts) where s = i + j.
    for (int64_t s = 2; s <= 2 * L - 2; ++s) {
        // Determine the range of row indices i that belong to this diagonal.
        int64_t i_min = s - (L - 1);
        if (i_min < 1) i_min = 1;
        int64_t i_max = s / 2; // floor(s/2) guarantees i <= j.
        if (i_max > L - 1) i_max = L - 1;
        int64_t diag_len = i_max - i_min + 1;
        if (diag_len <= 0) continue;

        // Process the diagonal in chunks that fit into the thread block.
        for (int64_t offset = 0; offset < diag_len; offset += blockSize) {
            int64_t idx = offset + tid; // index within the current diagonal.
            double left = 0.0;
            double up   = 0.0;
            if (idx < diag_len) {
                int64_t i = i_min + idx;
                int64_t j = s - i; // because i + j = s.
                double a_ij = a[i * L + j]; // original value of the element.

                if (s == 2) {
                    // Base case: dependencies lie outside the triangular region and are untouched.
                    left = a[i * L + (j - 1)];
                    up   = a[(i - 1) * L + j];
                } else {
                    // Dependencies are on the previous diagonal (s‑1). They are stored in 'prev'.
                    int64_t s_prev = s - 1;
                    int64_t i_min_prev = s_prev - (L - 1);
                    if (i_min_prev < 1) i_min_prev = 1;
                    int64_t i_max_prev = s_prev / 2;
                    if (i_max_prev > L - 1) i_max_prev = L - 1;
                    // Index of i in the previous diagonal, if it exists.
                    if (i >= i_min_prev && i <= i_max_prev) {
                        left = prev[i - i_min_prev];
                    } else {
                        // Outside the triangular region – read the original value.
                        left = a[i * L + (j - 1)];
                    }
                    if (i - 1 >= i_min_prev && i - 1 <= i_max_prev) {
                        up = prev[i - 1 - i_min_prev];
                    } else {
                        up = a[(i - 1) * L + j];
                    }
                }
                double new_val = a_ij + left + up;
                // Store the newly computed value in the current‑diagonal buffer.
                cur[idx] = new_val;
                // Write the result back to global memory.
                a[i * L + j] = new_val;
            }
            // Synchronise before the next chunk (or before the buffers are swapped).
            __syncthreads();
        }
        // Swap the buffers: current becomes previous for the next iteration.
        double *tmp = prev;
        prev = cur;
        cur = tmp;
    }
}

// Host entry point matching the C ABI expected by the benchmark harness.
extern "C" void wf_triangular_fp64(double * __restrict__ a, const int64_t LEN_2D) {
    const int64_t L = LEN_2D;
    // Choose a block size that is a multiple of warpSize and fits the shared memory budget.
    const int BLOCK_SIZE = 256; // 256 threads give a good balance.
    // Shared memory needed: two buffers of size ceil(L/2)+1 doubles.
    const size_t shared_bytes = (size_t)((L + 1) / 2 + 1) * 2 * sizeof(double);
    hipLaunchKernelGGL(wf_triangular_kernel,
                       dim3(1), dim3(BLOCK_SIZE), shared_bytes, 0,
                       a, L);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "HIP launch error: %s\n", hipGetErrorString(err));
        abort();
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "HIP sync error: %s\n", hipGetErrorString(err));
        abort();
    }
}
