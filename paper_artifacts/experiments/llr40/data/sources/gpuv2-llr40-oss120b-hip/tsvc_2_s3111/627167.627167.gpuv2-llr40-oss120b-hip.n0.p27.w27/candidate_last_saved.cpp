#include <hip/hip_runtime.h>
#include <cstdint>

// Forward declaration of the device kernel (implemented in device source)
__global__ void tsvc_2_s3111_kernel(const double * __restrict__ a, double * __restrict__ b, int64_t LEN_1D);

extern "C" void tsvc_2_s3111_fp64(const double * __restrict__ a, double * __restrict__ b, const int64_t LEN_1D) {
    // Initialize output accumulator to zero
    b[0] = 0.0;
    const int blockSize = 256; // power-of-two for reduction
    int gridSize = (LEN_1D + blockSize - 1) / blockSize;
    // Launch kernel with shared memory equal to blockSize * sizeof(double)
    hipLaunchKernelGGL(tsvc_2_s3111_kernel, dim3(gridSize), dim3(blockSize), blockSize * sizeof(double), 0,
                       a, b, LEN_1D);
    // Ensure kernel completion before returning
    hipDeviceSynchronize();
}
