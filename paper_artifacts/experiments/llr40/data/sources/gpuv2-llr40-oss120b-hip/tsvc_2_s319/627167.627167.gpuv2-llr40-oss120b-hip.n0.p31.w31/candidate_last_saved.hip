#include <hip/hip_runtime.h>
#include <cstdint>

// Device kernel performing vector additions and partial sum accumulation
extern "C" __global__ void tsvc_kernel(double * __restrict__ a, double * __restrict__ b,
                                         const double * __restrict__ c,
                                         const double * __restrict__ d,
                                         const double * __restrict__ e,
                                         int64_t N,
                                         double * __restrict__ sum_ptr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        double a_val = c[idx] + d[idx];
        a[idx] = a_val;
        double b_val = c[idx] + e[idx];
        b[idx] = b_val;
        double partial = a_val + b_val;
        // atomicAdd for double is supported on recent GPUs
        atomicAdd(sum_ptr, partial);
    }
}

// Exported symbol matching the reference ABI
extern "C" void tsvc_2_s319_fp64(double * __restrict__ a,
                                  double * __restrict__ b,
                                  const double * __restrict__ c,
                                  const double * __restrict__ d,
                                  const double * __restrict__ e,
                                  const int64_t LEN_1D) {
    // Choose launch configuration
    const int threads_per_block = 256;
    int blocks = (LEN_1D + threads_per_block - 1) / threads_per_block;
    if (blocks == 0) blocks = 1;

    // Allocate device memory for a single double to hold the sum
    double *d_sum;
    hipError_t err = hipMalloc(&d_sum, sizeof(double));
    if (err != hipSuccess) return; // early exit on OOM (unlikely in benchmark)
    // Initialize sum to zero
    double zero = 0.0;
    err = hipMemcpy(d_sum, &zero, sizeof(double), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        hipFree(d_sum);
        return;
    }

    // Launch the per-element kernel
    hipLaunchKernelGGL(tsvc_kernel, dim3(blocks), dim3(threads_per_block), 0, 0,
                       a, b, c, d, e, LEN_1D, d_sum);
    // Wait for kernel to finish
    err = hipGetLastError();
    if (err != hipSuccess) {
        hipFree(d_sum);
        return;
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        hipFree(d_sum);
        return;
    }

    // Copy back the sum to host
    double sum;
    err = hipMemcpy(&sum, d_sum, sizeof(double), hipMemcpyDeviceToHost);
    hipFree(d_sum);
    if (err != hipSuccess) {
        return;
    }

    // Write the sum to b[0] on the device (b points to device memory)
    err = hipMemcpy(b, &sum, sizeof(double), hipMemcpyHostToDevice);
    // ignore err, nothing else to do
    (void)err;
}
