/* Host part of TSVC s3110 HIP kernel */
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <limits>

// Helper macro for error checking – aborts on any HIP error.
#define HIP_CALL(call) do {\
    hipError_t err = (call);\
    if (err != hipSuccess) {\
        const char* errStr = hipGetErrorString(err);\
        fprintf(stderr, "HIP error: %s:%d: %s\n", __FILE__, __LINE__, errStr);\
        exit(1);\
    }\
} while (0)

constexpr int BLOCK_SIZE = 256;

// Forward declaration of the device kernel (defined in the device source).
__global__ void tsvc_2_s3110_kernel(const double* __restrict__ aa,
                                    const int64_t LEN_2D,
                                    double* __restrict__ partial_max,
                                    int64_t* __restrict__ partial_idx);

extern "C" void tsvc_2_s3110_fp64(const double* __restrict__ aa,
                                   double* __restrict__ bb,
                                   const int64_t LEN_2D) {
    constexpr int blockSize = BLOCK_SIZE;
    int64_t totalElements = LEN_2D * LEN_2D;
    int64_t numBlocks = (totalElements + blockSize - 1) / blockSize;
    if (numBlocks > 2147483647LL) numBlocks = 2147483647LL;

    double* d_partial_max = nullptr;
    int64_t* d_partial_idx = nullptr;
    HIP_CALL(hipMalloc(&d_partial_max, numBlocks * sizeof(double)));
    HIP_CALL(hipMalloc(&d_partial_idx, numBlocks * sizeof(int64_t)));

    // Launch the kernel.
    hipLaunchKernelGGL(tsvc_2_s3110_kernel,
                       dim3(static_cast<unsigned int>(numBlocks)),
                       dim3(blockSize), 0, 0,
                       aa, LEN_2D, d_partial_max, d_partial_idx);
    // Check for errors and synchronize.
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());

    // Retrieve partial results.
    std::vector<double> h_partial_max(numBlocks);
    std::vector<int64_t> h_partial_idx(numBlocks);
    HIP_CALL(hipMemcpy(h_partial_max.data(), d_partial_max, numBlocks * sizeof(double), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(h_partial_idx.data(), d_partial_idx, numBlocks * sizeof(int64_t), hipMemcpyDeviceToHost));

    // Final reduction on host.
    double maxv = -std::numeric_limits<double>::infinity();
    int64_t bestFlatIdx = -1;
    for (int64_t i = 0; i < numBlocks; ++i) {
        if (h_partial_max[i] > maxv) {
            maxv = h_partial_max[i];
            bestFlatIdx = h_partial_idx[i];
        }
    }

    int64_t xindex = bestFlatIdx / LEN_2D;
    int64_t yindex = bestFlatIdx % LEN_2D;
    double chksum = maxv + static_cast<double>(xindex) + static_cast<double>(yindex);

    // Write the checksum back to device memory.
    HIP_CALL(hipMemcpy(const_cast<double*>(bb), &chksum, sizeof(double), hipMemcpyHostToDevice));

    // Clean up.
    HIP_CALL(hipFree(d_partial_max));
    HIP_CALL(hipFree(d_partial_idx));
}

