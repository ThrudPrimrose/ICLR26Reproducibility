#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <cstddef>
#include <cstdint>

// Forward declarations of device kernels.
extern __global__ void compute_terms_kernel(const double *c, const double *d, const double *e,
                                            double *t, const int64_t N);
extern __global__ void update_ab_kernel(double *a, double *b, const double *c, const double *e,
                                        const double *t_scan, const int64_t N);

extern "C" void tsvc_2_s323_fp64(double *a, double *b,
                                 const double *c, const double *d,
                                 const double *e, const int64_t LEN_1D) {
    // Allocate temporary buffers for the per-index terms and their scan.
    double *t_in = nullptr;
    double *t_scan = nullptr;
    hipError_t err;
    err = hipMalloc(&t_in, static_cast<size_t>(LEN_1D) * sizeof(double));
    if (err != hipSuccess) return;
    err = hipMalloc(&t_scan, static_cast<size_t>(LEN_1D) * sizeof(double));
    if (err != hipSuccess) { hipFree(t_in); return; }

    // Launch kernel to compute t[i].
    const int blockSize = 256;
    const int gridSize = (static_cast<int>(LEN_1D) + blockSize - 1) / blockSize;
    dim3 block(blockSize);
    dim3 grid(gridSize);
    hipLaunchKernelGGL(compute_terms_kernel, grid, block, 0, 0,
                       c, d, e, t_in, LEN_1D);
    hipDeviceSynchronize();

    // Perform inclusive scan on t_in to produce t_scan.
    size_t temp_storage_bytes = 0;
    hipcub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes,
                                     t_in, t_scan, static_cast<int>(LEN_1D));
    void *d_temp_storage = nullptr;
    err = hipMalloc(&d_temp_storage, temp_storage_bytes);
    if (err != hipSuccess) {
        hipFree(t_in);
        hipFree(t_scan);
        return;
    }
    hipError_t scan_err = hipcub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                                            t_in, t_scan, static_cast<int>(LEN_1D));
    // Free temporary storage used by the scan.
    hipFree(d_temp_storage);
    hipDeviceSynchronize();

    // Launch kernel to write the final a and b values.
    hipLaunchKernelGGL(update_ab_kernel, grid, block, 0, 0,
                       a, b, c, e, t_scan, LEN_1D);
    hipDeviceSynchronize();

    // Clean up temporary buffers.
    hipFree(t_in);
    hipFree(t_scan);
}
