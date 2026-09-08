#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

extern "C" __global__ void ext_war_unit_kernel(double *a, const double *b, const double *tmp, int64_t n);

extern "C" void ext_war_unit_fp64(double *__restrict__ a, const double *__restrict__ b, const int64_t LEN_1D) {
    double *tmp;
    hipMalloc(&tmp, static_cast<size_t>(LEN_1D) * sizeof(double));
    hipMemcpy(tmp, a, static_cast<size_t>(LEN_1D) * sizeof(double), hipMemcpyDeviceToDevice);
    int64_t n = LEN_1D - 1;
    const int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    hipLaunchKernelGGL(ext_war_unit_kernel, dim3(blocks), dim3(threads), 0, 0, a, b, tmp, n);
    hipDeviceSynchronize();
    hipFree(tmp);
}

extern "C" __global__ void ext_war_unit_kernel(double *a, const double *b, const double *tmp, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = tmp[idx + 1] + b[idx];
    }
}
