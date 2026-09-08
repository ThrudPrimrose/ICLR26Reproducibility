#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdio>

// Helper macro for error checking
inline void hipCheck(hipError_t err, const char* file, int line) {
    if (err != hipSuccess) {
        fprintf(stderr, "HIP error %s:%d: %s\n", file, line, hipGetErrorString(err));
        abort();
    }
}
#define CHECKHIP(call) hipCheck((call), __FILE__, __LINE__)

// Kernel to compute mask: mask[i] = (src[i] > 0) ? 1 : 0
__global__ void computeMaskKernel(const double* __restrict__ src,
                                   unsigned int* __restrict__ mask,
                                   int64_t N) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        mask[idx] = (src[idx] > 0.0) ? 1u : 0u;
    }
}

// Kernel to scatter results using the exclusive scan indices
__global__ void scatterKernel(const double* __restrict__ src,
                              const double* __restrict__ weight,
                              const unsigned int* __restrict__ mask,
                              const unsigned int* __restrict__ indices,
                              double* __restrict__ packed,
                              int64_t N) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (mask[idx]) {
            packed[indices[idx]] = src[idx] * weight[idx];
        }
    }
}

// Kernel to write the final count (sum of mask) to out_count[0]
__global__ void writeCountKernel(const unsigned int* __restrict__ mask,
                                 const unsigned int* __restrict__ indices,
                                 int64_t* __restrict__ out_count,
                                 int64_t N) {
    if (N == 0) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            out_count[0] = 0;
        }
        return;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned int total = indices[N - 1] + mask[N - 1];
        out_count[0] = static_cast<int64_t>(total);
    }
}

extern "C" void compact_threshold_pack(const double* __restrict__ src,
                                        const double* __restrict__ weight,
                                        double* __restrict__ packed,
                                        int64_t* __restrict__ out_count,
                                        const int64_t LEN_1D) {
    // Handle empty input
    if (LEN_1D <= 0) {
        if (out_count) out_count[0] = 0;
        return;
    }

    const unsigned int blockSize = 256U;
    const unsigned int gridSize = static_cast<unsigned int>((LEN_1D + blockSize - 1) / blockSize);
    hipError_t status;

    // Allocate device temporary buffers: mask and exclusive-scan indices
    unsigned int* d_mask = nullptr;
    unsigned int* d_indices = nullptr;
    CHECKHIP(hipMalloc(&d_mask, static_cast<std::size_t>(LEN_1D) * sizeof(unsigned int)));
    CHECKHIP(hipMalloc(&d_indices, static_cast<std::size_t>(LEN_1D) * sizeof(unsigned int)));

    // Compute mask
    computeMaskKernel<<<gridSize, blockSize>>>(src, d_mask, LEN_1D);
    CHECKHIP(hipGetLastError());
    CHECKHIP(hipDeviceSynchronize());

    // Determine temporary storage size for exclusive sum
    void* d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    status = hipcub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_mask, d_indices, static_cast<std::size_t>(LEN_1D));
    CHECKHIP(status);
    CHECKHIP(hipMalloc(&d_temp_storage, temp_storage_bytes));

    // Perform exclusive sum (indices = prefix sum of mask)
    status = hipcub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_mask, d_indices, static_cast<std::size_t>(LEN_1D));
    CHECKHIP(status);
    CHECKHIP(hipGetLastError());
    CHECKHIP(hipDeviceSynchronize());

    // Scatter results
    scatterKernel<<<gridSize, blockSize>>>(src, weight, d_mask, d_indices, packed, LEN_1D);
    CHECKHIP(hipGetLastError());
    CHECKHIP(hipDeviceSynchronize());

    // Write final count to out_count
    writeCountKernel<<<1, 1>>>(d_mask, d_indices, out_count, LEN_1D);
    CHECKHIP(hipGetLastError());
    CHECKHIP(hipDeviceSynchronize());

    // Clean up
    CHECKHIP(hipFree(d_mask));
    CHECKHIP(hipFree(d_indices));
    CHECKHIP(hipFree(d_temp_storage));
}
