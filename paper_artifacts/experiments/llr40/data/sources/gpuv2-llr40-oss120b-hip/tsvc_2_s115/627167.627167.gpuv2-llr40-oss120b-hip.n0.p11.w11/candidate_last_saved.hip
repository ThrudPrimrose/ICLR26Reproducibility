#include <hip/hip_runtime.h>
#include <cstdint>

// Kernel for inner loop of tsvc_2_s115: updates a[i] for i > j
__global__ void s115_kernel(const int64_t N, const int64_t j, const double a_j, double *a, const double *aa) {
    int64_t idx = j + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Compute linear index for aa[j][idx]
        // Row-major: aa[j * N + idx]
        a[idx] -= aa[j * N + idx] * a_j;
    }
}

extern "C" void tsvc_2_s115_fp64(double *__restrict__ a, const double *__restrict__ aa, const int64_t LEN_2D) {
    // Choose a reasonable block size
    const int blockSize = 256;
    for (int64_t j = 0; j < LEN_2D; ++j) {
        // Load a[j] on host after previous kernels have finished
        double a_j = a[j];
        int64_t start = j + 1;
        int64_t count = LEN_2D - start;
        if (count <= 0) {
            // No more i values for this j
            continue;
        }
        int64_t gridSize = (count + blockSize - 1) / blockSize;
        // Launch kernel to handle i in [j+1, LEN_2D)
        hipLaunchKernelGGL(s115_kernel, dim3(gridSize), dim3(blockSize), 0, 0,
                           LEN_2D, j, a_j, a, aa);
        // Ensure kernel completion before next iteration, to preserve ordering
        hipDeviceSynchronize();
    }
}
