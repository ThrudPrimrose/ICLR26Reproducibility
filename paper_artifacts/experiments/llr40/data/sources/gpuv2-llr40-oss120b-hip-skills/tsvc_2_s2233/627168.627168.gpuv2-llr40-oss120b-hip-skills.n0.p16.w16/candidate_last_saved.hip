// HIP implementation of TSVC 2 s2233 microkernel (fp64).
// Signature matches the C reference: void tsvc_2_s2233_fp64(double *restrict aa,
// double *restrict bb, const double *restrict cc, const int64_t LEN_2D)
//
// The kernel computes column-wise prefix sums for aa and row-wise prefix sums for bb.
// Each column (for aa) and each row (for bb) are processed independently by a thread.
// This simple implementation uses one thread per column/row and loops sequentially,
// which provides massive parallelism across columns/rows despite the inherent data
// dependencies within each scan.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

// Kernel for scanning column 'i' of array aa
__global__ void scan_aa_kernel(double* __restrict__ aa, const double* __restrict__ cc, int64_t LEN_2D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 8; // start from i=8
    if (i >= LEN_2D) return;
    // Perform the cumulative sum down the column.
    for (int64_t j = 8; j < LEN_2D; ++j) {
        // linear index for element (j,i) in row-major layout
        int64_t idx_cur = j * LEN_2D + i;
        int64_t idx_prev = (j - 1) * LEN_2D + i;
        aa[idx_cur] = aa[idx_prev] + cc[idx_cur];
    }
}

// Kernel for scanning row 'i' of array bb
__global__ void scan_bb_kernel(double* __restrict__ bb, const double* __restrict__ cc, int64_t LEN_2D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 8; // start from i=8
    if (i >= LEN_2D) return;
    // Perform the cumulative sum across the row.
    for (int64_t j = 8; j < LEN_2D; ++j) {
        int64_t idx_cur = i * LEN_2D + j;
        int64_t idx_prev = i * LEN_2D + (j - 1);
        bb[idx_cur] = bb[idx_prev] + cc[idx_cur];
    }
}

extern "C" void tsvc_2_s2233_fp64(double* __restrict__ aa, double* __restrict__ bb,
                                   const double* __restrict__ cc, const int64_t LEN_2D) {
    const int threads = 256;
    int64_t work = LEN_2D - 8;
    if (work < 0) work = 0;
    int blocks = (static_cast<int>((work + threads - 1) / threads));

    // Launch column scan for aa
    hipLaunchKernelGGL(scan_aa_kernel,
                       dim3(blocks), dim3(threads), 0, 0,
                       aa, cc, LEN_2D);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        const char* msg = hipGetErrorString(err);
        fprintf(stderr, "HIP error launching scan_aa_kernel: %s\n", msg);
        abort();
    }

    // Launch row scan for bb
    hipLaunchKernelGGL(scan_bb_kernel,
                       dim3(blocks), dim3(threads), 0, 0,
                       bb, cc, LEN_2D);
    err = hipGetLastError();
    if (err != hipSuccess) {
        const char* msg = hipGetErrorString(err);
        fprintf(stderr, "HIP error launching scan_bb_kernel: %s\n", msg);
        abort();
    }

    // Ensure kernels are complete before returning.
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        const char* msg = hipGetErrorString(err);
        fprintf(stderr, "HIP error after synchronization: %s\n", msg);
        abort();
    }
}
