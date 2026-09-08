#include <hip/hip_runtime.h>
#include <cstdint>

// Device kernel: find first element > K (hardcoded K = 1.0) and store its index and value.
// Uses atomicMin to keep track of smallest index seen.

extern "C" __global__ void ext_break_capture_kernel(const double *a, int64_t *out_index, double *out_value, int64_t LEN_1D) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= LEN_1D) return;
    double val = a[idx];
    const double K = 1.0;
    if (val > K) {
        // Perform atomic min on index (treated as unsigned long long).
        unsigned long long *idx_ptr = reinterpret_cast<unsigned long long*>(out_index);
        unsigned long long old = atomicMin(idx_ptr, static_cast<unsigned long long>(idx));
        // Only the thread that successfully lowered the minimum writes the result.
        if (static_cast<unsigned long long>(idx) < old) {
            out_index[0] = idx;      // write the discovered index
            out_value[0] = val;      // capture the value at that index
        }
    }
}

// Host wrapper with C ABI expected by the benchmark harness.
extern "C" void ext_break_capture_fp64(const double *a, int64_t *out_index, double *out_value, const int64_t LEN_1D) {
    // Initialize output buffers on device: out_index = -1, out_value = -1.0
    int64_t sentinel = -1; // all bits 1, works with atomicMin as initial large value
    double sentinel_val = -1.0;
    hipMemcpy((void*)out_index, &sentinel, sizeof(int64_t), hipMemcpyHostToDevice);
    hipMemcpy((void*)out_value, &sentinel_val, sizeof(double), hipMemcpyHostToDevice);

    // Launch kernel
    const int blockSize = 256;
    const int gridSize = static_cast<int>((LEN_1D + blockSize - 1) / blockSize);
    hipLaunchKernelGGL(ext_break_capture_kernel,
                       dim3(gridSize), dim3(blockSize), 0, 0,
                       a, out_index, out_value, LEN_1D);
    hipDeviceSynchronize();
}

