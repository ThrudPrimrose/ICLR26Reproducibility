#include <hip/hip_runtime.h>
#include <cstdint>

// Kernel to compute array a
extern "C" __global__ void compute_a_kernel(double* a, const double* b, const double* c, int64_t N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N - 1) { // match reference: only compute up to N-2
        double bi = b[i];
        double ci = c[i];
        a[i] = bi + ci * ci + bi * bi + ci;
    }
}

// Kernel to compute array d using a (a[i+1] may be zero if not set)
extern "C" __global__ void compute_d_kernel(const double* a, double* d, int64_t N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N - 1) {
        d[i] = a[i] + a[i + 1];
    }
}

// Host wrapper that launches the kernels
extern "C" void tsvc_2_s1244_fp64(double * __restrict__ a, const double * __restrict__ b, const double * __restrict__ c,
                                   double * __restrict__ d, const int64_t LEN_1D) {
    const int threadsPerBlock = 256;
    int blocks = (LEN_1D + threadsPerBlock - 1) / threadsPerBlock;
    // Launch kernel to compute a (note: compute a only up to LEN_1D-1 as reference)
    hipLaunchKernelGGL(compute_a_kernel, dim3(blocks), dim3(threadsPerBlock), 0, 0, a, b, c, LEN_1D);
    hipDeviceSynchronize(); // ensure a is ready before computing d
    // Launch kernel to compute d
    hipLaunchKernelGGL(compute_d_kernel, dim3(blocks), dim3(threadsPerBlock), 0, 0, a, d, LEN_1D);
    hipDeviceSynchronize();
}
