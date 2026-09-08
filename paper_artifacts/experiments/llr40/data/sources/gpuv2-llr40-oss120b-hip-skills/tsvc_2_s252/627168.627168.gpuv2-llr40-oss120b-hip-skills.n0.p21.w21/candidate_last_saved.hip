#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <cstdint>
#include <cstdio>

inline void hipCheck(hipError_t err) {
    if (err != hipSuccess) {
        fprintf(stderr, "HIP error: %s\n", hipGetErrorString(err));
        std::abort();
    }
}

__global__ void productKernel(const double* __restrict__ b, const double* __restrict__ c, double* __restrict__ out, int64_t n) {
    int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (idx < n) {
        out[idx] = b[idx] * c[idx];
    }
}

__global__ void finalKernel(const double* __restrict__ d, const double* __restrict__ prefix, double* __restrict__ a, int64_t n) {
    int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (idx < n) {
        a[idx] = d[idx] + prefix[idx];
    }
}

extern "C" void tsvc_2_s252_fp64(double* __restrict__ a, const double* __restrict__ b, const double* __restrict__ c, const int64_t LEN_1D) {
    int64_t n = LEN_1D;
    if (n <= 0) return;
    // Allocate temporary storage for product and prefix
    double* d = nullptr;
    double* prefix = nullptr;
    hipCheck(hipMalloc(&d, n * sizeof(double)));
    hipCheck(hipMalloc(&prefix, n * sizeof(double)));

    const int blockSize = 256;
    int64_t gridSize = (n + blockSize - 1) / blockSize;
    // Compute product
    productKernel<<<gridSize, blockSize>>>(b, c, d, n);
    hipCheck(hipGetLastError());
    hipCheck(hipDeviceSynchronize());

    // Determine temporary storage size for exclusive sum
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    hipCheck(hipcub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d, prefix, static_cast<size_t>(n)));
    // Allocate temporary storage
    hipCheck(hipMalloc(&d_temp_storage, temp_storage_bytes));
    // Perform exclusive sum (prefix)
    hipCheck(hipcub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d, prefix, static_cast<size_t>(n)));
    hipCheck(hipGetLastError());
    hipCheck(hipDeviceSynchronize());
    // Free temporary storage
    hipCheck(hipFree(d_temp_storage));
    // Compute final a = d + prefix
    finalKernel<<<gridSize, blockSize>>>(d, prefix, a, n);
    hipCheck(hipGetLastError());
    hipCheck(hipDeviceSynchronize());
    // Free temporary buffers
    hipCheck(hipFree(d));
    hipCheck(hipFree(prefix));
}

