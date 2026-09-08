#include <hip/hip_runtime.h>
#include <cstdint>

// Kernel for computing prefix sum over rows for each column of aa.
// Each thread processes one column (i), performing a sequential scan
// over the rows (j) for that column.
__global__ void kernel_aa(double *aa, const double *cc, int64_t LEN) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t i = idx + 8; // column index, start at 8 as per reference
    if (i >= LEN) return;
    // The recurrence starts at row j = 8.
    // Use the value at row 7 as the initial value for the scan.
    int64_t base_prev = (8 - 1) * LEN + i; // row 7, column i
    double prev = aa[base_prev];
    for (int64_t j = 8; j < LEN; ++j) {
        int64_t off = j * LEN + i;
        prev += cc[off];
        aa[off] = prev;
    }
}

// Kernel for computing prefix sum over rows for each column of bb (i dimension).
// Here each thread processes one column of the matrix (j), performing a sequential
// scan over the rows (i) for that column.
__global__ void kernel_bb(double *bb, const double *cc, int64_t LEN) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t j = idx + 8; // column index, start at 8
    if (j >= LEN) return;
    // Initial value comes from row 7.
    int64_t base_prev = (8 - 1) * LEN + j; // row 7, column j
    double prev = bb[base_prev];
    for (int64_t i = 8; i < LEN; ++i) {
        int64_t off = i * LEN + j;
        prev += cc[off];
        bb[off] = prev;
    }
}

extern "C" void tsvc_2_s2233_fp64(double * __restrict__ aa,
                                   double * __restrict__ bb,
                                   const double * __restrict__ cc,
                                   const int64_t LEN_2D) {
    // Decide thread block size. 256 is a reasonable default.
    const int threads = 256;
    int64_t cols = LEN_2D - 8;
    if (cols <= 0) return;
    dim3 block(threads);
    dim3 grid((cols + threads - 1) / threads);
    // Launch kernels for aa and bb.
    kernel_aa<<<grid, block>>>(aa, cc, LEN_2D);
    kernel_bb<<<grid, block>>>(bb, cc, LEN_2D);
    // Ensure kernels have finished before returning.
    hipDeviceSynchronize();
}
