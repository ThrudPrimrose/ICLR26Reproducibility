// HIP implementation of TSVC tsvc_2 kernel s233 (fp64)
// Exported symbol matches reference: tsvc_2_s233_fp64
// The kernel computes two independent recurrences:
//   aa[j,i] = aa[j-1,i] + cc[j,i]   (dependency along j for each column i)
//   bb[j,i] = bb[j,i-1] + cc[j,i]   (dependency along i for each row j)
// We launch two separate kernels to parallelize across the independent dimension.

#include <hip/hip_runtime.h>
#include <cstdint>

// Define restrict for C++ compatibility if not already defined
#ifndef __HIPCC__
#define __restrict__ __restrict__
#endif

// Kernel for computing aa column-wise recurrence.
__global__ void tsvc_2_s233_aa_kernel(double *__restrict__ aa, const double *__restrict__ cc, int64_t N) {
    // Global column index i (starting from 8)
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x + 8;
    if (i >= N) return;
    // Sequential loop over rows j
    for (int64_t j = 8; j < N; ++j) {
        // aa[j,i] = aa[j-1,i] + cc[j,i]
        aa[j * N + i] = aa[(j - 1) * N + i] + cc[j * N + i];
    }
}

// Kernel for computing bb row-wise recurrence.
__global__ void tsvc_2_s233_bb_kernel(double *__restrict__ bb, const double *__restrict__ cc, int64_t N) {
    // Global row index j (starting from 8)
    int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x + 8;
    if (j >= N) return;
    for (int64_t i = 8; i < N; ++i) {
        bb[j * N + i] = bb[j * N + (i - 1)] + cc[j * N + i];
    }
}

extern "C" void tsvc_2_s233_fp64(double *__restrict__ aa,
                                 double *__restrict__ bb,
                                 const double *__restrict__ cc,
                                 const int64_t LEN_2D) {
    // If the effective range is empty, nothing to do.
    if (LEN_2D <= 8) return;
    // Choose block size (tuned for typical GPU). 256 threads per block is reasonable.
    const int blockSize = 256;
    // Launch aa kernel: one thread per column i.
    int64_t cols = LEN_2D - 8;
    int64_t gridAA = (cols + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(tsvc_2_s233_aa_kernel,
                       dim3(gridAA), dim3(blockSize), 0, 0,
                       aa, cc, LEN_2D);
    // Launch bb kernel: one thread per row j.
    int64_t rows = LEN_2D - 8;
    int64_t gridBB = (rows + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(tsvc_2_s233_bb_kernel,
                       dim3(gridBB), dim3(blockSize), 0, 0,
                       bb, cc, LEN_2D);
    // Ensure kernels have completed before returning.
    hipDeviceSynchronize();
}

