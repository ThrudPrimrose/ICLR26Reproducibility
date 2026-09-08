#include <hip/hip_runtime.h>
#include <cstdint>

// Kernel that computes aa[i][j] = aa[i-1][j-1] + bb[i][j] for i,j >= 1.
// The input arrays are provided as flat row-major 2D arrays of size LEN_2D x LEN_2D.
// aa and bb reside in device memory; the first row and first column of aa are
// used as base values and are not overwritten.
extern "C" __global__ void tsvc_2_s119_fp64_kernel(double * __restrict__ aa,
                                                    const double * __restrict__ bb,
                                                    const int64_t LEN_2D) {
    // Total number of diagonals (including length‑1 diagonals).
    const int64_t total_diags = 2 * LEN_2D - 1;
    int64_t diag = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; // diag index in [0,total_diags)
    if (diag >= total_diags) return;

    // Map linear diag index to offset d = i - j.
    // The central diagonal (i == j) has d = 0 and index (LEN_2D - 1).
    int64_t d = diag - (LEN_2D - 1);
    int64_t absd = d >= 0 ? d : -d;
    int64_t length = LEN_2D - absd; // number of elements on this diagonal
    if (length <= 1) return; // no interior elements to update

    // Compute base index: the element just before the first interior element on this diagonal.
    int64_t base_idx;
    if (d >= 0) {
        // Base is aa[d][0]
        base_idx = d * LEN_2D; // row d, column 0
    } else {
        // d < 0: Base is aa[0][-d]
        base_idx = -d; // row 0, column -d
    }
    // Load the base value (remains unchanged by this kernel).
    double sum = aa[base_idx];

    // Stride between consecutive elements on the same diagonal in the flat array.
    const int64_t stride = LEN_2D + 1;
    // Index of the first interior element (i >= 1 && j >= 1).
    int64_t idx = base_idx + stride;
    // Number of interior elements to compute on this diagonal.
    int64_t interior = length - 1;
    for (int64_t step = 0; step < interior; ++step) {
        sum += bb[idx];
        aa[idx] = sum;
        idx += stride;
    }
}

extern "C" void tsvc_2_s119_fp64(double * __restrict__ aa,
                                 const double * __restrict__ bb,
                                 const int64_t LEN_2D) {
    // Launch one thread per diagonal.
    const int threads_per_block = 256;
    int64_t total_diags = 2 * LEN_2D - 1;
    int64_t blocks = (total_diags + threads_per_block - 1) / threads_per_block;
    hipLaunchKernelGGL(tsvc_2_s119_fp64_kernel,
                       dim3(blocks), dim3(threads_per_block), 0, 0,
                       aa, bb, LEN_2D);
    hipDeviceSynchronize();
}

