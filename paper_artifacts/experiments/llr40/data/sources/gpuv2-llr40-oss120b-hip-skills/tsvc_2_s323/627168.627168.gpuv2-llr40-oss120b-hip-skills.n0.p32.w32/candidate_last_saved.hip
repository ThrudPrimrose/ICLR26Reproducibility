#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <cstdint>
#include <cstdio>

// Macro to check HIP API calls and abort on error.
#define HIP_CHECK(err)                                                          \
    do {                                                                        \
        hipError_t _err = (err);                                                \
        if (_err != hipSuccess) {                                               \
            fprintf(stderr, "HIP error: %s at %s:%d\n", hipGetErrorString(_err), __FILE__, __LINE__); \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// Forward declaration of device kernels defined in the device source.
extern "C" __global__ void compute_g_kernel(const double* __restrict__ c,
                                            const double* __restrict__ d,
                                            const double* __restrict__ e,
                                            double* __restrict__ g,
                                            int64_t LEN_1D);
extern "C" __global__ void final_kernel(double* __restrict__ a,
                                        double* __restrict__ b,
                                        const double* __restrict__ c,
                                        const double* __restrict__ d,
                                        const double* __restrict__ e,
                                        const double* __restrict__ prefix,
                                        const double* __restrict__ g,
                                        int64_t LEN_1D);

// Exported entry point with C linkage.
extern "C" void tsvc_2_s323_fp64(double* __restrict__ a,
                                 double* __restrict__ b,
                                 const double* __restrict__ c,
                                 const double* __restrict__ d,
                                 const double* __restrict__ e,
                                 const int64_t LEN_1D) {
    if (LEN_1D <= 1) return;

    int64_t N = LEN_1D - 1;
    const int blockSize = 256;
    int gridSize = static_cast<int>((LEN_1D + blockSize - 1) / blockSize);

    // Allocate temporary device buffers.
    double* d_g = nullptr;
    double* d_prefix = nullptr;
    HIP_CHECK(hipMalloc(&d_g, sizeof(double) * static_cast<size_t>(LEN_1D)));
    HIP_CHECK(hipMalloc(&d_prefix, sizeof(double) * static_cast<size_t>(N)));

    // Compute g on the device.
    hipLaunchKernelGGL(compute_g_kernel,
                       dim3(gridSize), dim3(blockSize), 0, 0,
                       c, d, e, d_g, LEN_1D);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Prepare for exclusive scan of g[1..] (length N).
    size_t temp_storage_bytes = 0;
    HIP_CHECK(hipcub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes,
                                               d_g + 1, d_prefix, static_cast<int>(N)));
    void* d_temp_storage = nullptr;
    size_t storage = temp_storage_bytes > 0 ? temp_storage_bytes : 1;
    HIP_CHECK(hipMalloc(&d_temp_storage, storage));

    // Run the exclusive scan.
    HIP_CHECK(hipcub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                               d_g + 1, d_prefix, static_cast<int>(N)));
    HIP_CHECK(hipGetLastError());

    // Compute final a and b using the prefix sums.
    hipLaunchKernelGGL(final_kernel,
                       dim3(gridSize), dim3(blockSize), 0, 0,
                       a, b, c, d, e, d_prefix, d_g, LEN_1D);
    HIP_CHECK(hipGetLastError());

    // Clean up temporary resources.
    HIP_CHECK(hipFree(d_g));
    HIP_CHECK(hipFree(d_prefix));
    HIP_CHECK(hipFree(d_temp_storage));

    // Synchronize before returning.
    HIP_CHECK(hipDeviceSynchronize());
}

