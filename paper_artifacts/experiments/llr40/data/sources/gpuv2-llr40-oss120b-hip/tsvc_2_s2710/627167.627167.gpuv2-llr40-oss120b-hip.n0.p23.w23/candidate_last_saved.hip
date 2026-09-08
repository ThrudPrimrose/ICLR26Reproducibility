#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

// Device kernel implementing the loop
static __global__ void tsvc_2_s2710_kernel(double * __restrict__ a,
                                            double * __restrict__ b,
                                            double * __restrict__ c,
                                            const double * __restrict__ d,
                                            const double * __restrict__ e,
                                            double x0,
                                            int64_t LEN_1D) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= LEN_1D) return;
    double ai = a[i];
    double bi = b[i];
    double di = d[i];
    double ei = e[i];
    if (ai > bi) {
        // a[i] += b[i] * d[i];
        a[i] = ai + bi * di;
        if (LEN_1D > 10) {
            // c[i] += d[i] * d[i];
            c[i] = c[i] + di * di;
        } else {
            // c[i] = d[i] * e[i] + 1.0;
            c[i] = di * ei + 1.0;
        }
    } else {
        // b[i] = a[i] + e[i] * e[i];
        b[i] = ai + ei * ei;
        if (x0 > 0.0) {
            // c[i] = a[i] + d[i] * d[i];
            c[i] = ai + di * di;
        } else {
            // c[i] += e[i] * e[i];
            c[i] = c[i] + ei * ei;
        }
    }
}

extern "C" void tsvc_2_s2710_fp64(double * __restrict__ a,
                                   double * __restrict__ b,
                                   double * __restrict__ c,
                                   const double * __restrict__ d,
                                   const double * __restrict__ e,
                                   const double * __restrict__ x,
                                   const int64_t LEN_1D) {
    const int blockSize = 256;
    const int gridSize = (int)((LEN_1D + blockSize - 1) / blockSize);
    // Load x[0] once for all threads
    double x0 = x[0];
    // Launch kernel
    hipLaunchKernelGGL(tsvc_2_s2710_kernel,
                       dim3(gridSize), dim3(blockSize), 0, 0,
                       a, b, c, d, e, x0, LEN_1D);
    // No need to synchronize; caller will handle.
}
