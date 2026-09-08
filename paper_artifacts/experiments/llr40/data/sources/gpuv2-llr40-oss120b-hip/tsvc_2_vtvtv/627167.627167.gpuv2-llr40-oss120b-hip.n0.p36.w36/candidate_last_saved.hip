#include <hip/hip_runtime.h>
#include <cstdint>

// Kernel that multiplies each element: a[i] = a[i] * b[i] * c[i]
// Using 64-bit integer length and restrict qualifier for pointers.
extern "C" __global__ void tsvc_2_vtvtv_kernel(const double* __restrict__ a_in,
                                                const double* __restrict__ b,
                                                const double* __restrict__ c,
                                                double* __restrict__ a_out,
                                                int64_t len) {
    // Use 64-bit index to avoid overflow for large arrays
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < len; i += stride) {
        // Load from a_in, b, c, compute product and store to a_out
        double av = a_in[i];
        double bv = b[i];
        double cv = c[i];
        a_out[i] = av * bv * cv;
    }
}

// Alternate kernel using vectorized double4 loads/stores for better memory throughput.
extern "C" __global__ void tsvc_2_vtvtv_kernel_vec4(const double* __restrict__ a_in,
                                                    const double* __restrict__ b,
                                                    const double* __restrict__ c,
                                                    double* __restrict__ a_out,
                                                    int64_t len) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    // Process 4 elements per iteration when possible
    int64_t i = idx * 4;
    int64_t stride4 = stride * 4;
    for (; i + 3 < len; i += stride4) {
        // Load four doubles at once
        double4 av = reinterpret_cast<const double4*>(a_in)[i / 4];
        double4 bv = reinterpret_cast<const double4*>(b)[i / 4];
        double4 cv = reinterpret_cast<const double4*>(c)[i / 4];
        // Compute elementwise product
        av.x = av.x * bv.x * cv.x;
        av.y = av.y * bv.y * cv.y;
        av.z = av.z * bv.z * cv.z;
        av.w = av.w * bv.w * cv.w;
        // Store back
        reinterpret_cast<double4*>(a_out)[i / 4] = av;
    }
    // Remaining elements (if any)
    for (int64_t j = i; j < len; ++j) {
        double av = a_in[j];
        double bv = b[j];
        double cv = c[j];
        a_out[j] = av * bv * cv;
    }
}

// Host entry point matching the C ABI expected by the benchmark harness.
extern "C" void tsvc_2_vtvtv_fp64(double * __restrict__ a, const double * __restrict__ b, const double * __restrict__ c, const int64_t LEN_1D) {
    // The kernel updates a in-place. We'll use the vectorized kernel for large sizes,
    // falling back to the scalar kernel for very small workloads.
    const int blockSize = 256; // Typical warp multiple
    // Compute grid size ensuring we cover all elements, using ceil division.
    int gridSize = static_cast<int>((LEN_1D + blockSize - 1) / blockSize);
    // Cap grid size to a reasonable maximum (max grid dim is 2147483647 for HIP, but we keep it modest).
    if (gridSize > 65535) gridSize = 65535;
    // Choose which kernel to launch based on vector length.
    if (LEN_1D >= 4 * blockSize) {
        // Launch vectorized kernel
        hipLaunchKernelGGL(tsvc_2_vtvtv_kernel_vec4,
                           dim3(gridSize), dim3(blockSize), 0, 0,
                           a, b, c, a, LEN_1D);
    } else {
        // Launch scalar kernel for small sizes
        hipLaunchKernelGGL(tsvc_2_vtvtv_kernel,
                           dim3(gridSize), dim3(blockSize), 0, 0,
                           a, b, c, a, LEN_1D);
    }
    // Ensure kernel execution completes before returning.
    hipDeviceSynchronize();
}
