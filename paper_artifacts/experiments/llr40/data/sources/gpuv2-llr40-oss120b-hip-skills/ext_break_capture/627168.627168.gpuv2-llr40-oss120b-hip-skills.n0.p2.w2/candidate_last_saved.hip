#include <hip/hip_runtime.h>
#include <cstdint>

// Device kernel to find the minimal index i where a[i] > 1.0.
// It updates out_index atomically with the smallest i found.
__global__ void ext_break_capture_find_kernel(const double* __restrict__ a,
                                              unsigned long long* __restrict__ out_index,
                                              const int64_t LEN_1D) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= LEN_1D) return;
    if (a[idx] > 1.0) {
        atomicMin(out_index, static_cast<unsigned long long>(idx));
    }
}

// Finalizer kernel: reads the index and writes output values.
__global__ void ext_break_capture_finalize_kernel(const double* __restrict__ a,
                                                  int64_t* __restrict__ out_index,
                                                  double* __restrict__ out_value,
                                                  const int64_t LEN_1D) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int64_t idx = out_index[0];
    if (idx >= LEN_1D) {
        out_index[0] = -1;
        out_value[0] = -1.0;
    } else {
        out_value[0] = a[idx];
    }
}

// Host entry point
extern "C" void ext_break_capture_fp64(const double* __restrict__ a,
                                       int64_t* __restrict__ out_index,
                                       double* __restrict__ out_value,
                                       const int64_t LEN_1D) {
    // Sentinel value: any index >= LEN_1D indicates not found.
    int64_t sentinel = LEN_1D;
    // Initialize out_index on device with sentinel.
    hipMemcpy(out_index, &sentinel, sizeof(int64_t), hipMemcpyHostToDevice);
    // Initialize out_value with -1.0.
    double neg_one = -1.0;
    hipMemcpy(out_value, &neg_one, sizeof(double), hipMemcpyHostToDevice);
    // Launch find kernel.
    const int blockSize = 256;
    int gridSize = (LEN_1D + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(ext_break_capture_find_kernel,
                       dim3(gridSize), dim3(blockSize), 0, 0,
                       a, reinterpret_cast<unsigned long long*>(out_index), LEN_1D);
    hipDeviceSynchronize();
    // Launch finalizer kernel (single thread) to write the results.
    hipLaunchKernelGGL(ext_break_capture_finalize_kernel,
                       dim3(1), dim3(1), 0, 0,
                       a, out_index, out_value, LEN_1D);
    hipDeviceSynchronize();
}
