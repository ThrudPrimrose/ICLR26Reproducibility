#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

#define HIP_CHECK(expr) do { \
    hipError_t _e = (expr); \
    if (_e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e), __FILE__, __LINE__); \
        abort(); \
    } \
} while (0)

__global__ void vpvts_vec_kernel(double4* __restrict__ a, const double4* __restrict__ b,
                                 const long n4, const double S) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long stride = (long)gridDim.x * blockDim.x;
    for (long j = i; j < n4; j += stride) {
        double4 bv = b[j];
        double4 av = a[j];
        av.x += bv.x * S;
        av.y += bv.y * S;
        av.z += bv.z * S;
        av.w += bv.w * S;
        a[j] = av;
    }
}

__global__ void vpvts_tail_kernel(double* __restrict__ a, const double* __restrict__ b,
                                  const long start, const long n, const double S) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[start + i] += b[start + i] * S;
    }
}

extern "C" void tsvc_2_vpvts_fp64(double* a, const double* b,
                                  const int64_t LEN_1D, const int64_t S) {
    const long n = (long)LEN_1D;
    const long n4 = n / 4;
    const long tail = n - n4 * 4;
    const int threads = 256;
    long blocks = (n4 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    if (n4 > 0) {
        hipLaunchKernelGGL(vpvts_vec_kernel, dim3((unsigned)blocks), dim3(threads), 0, 0,
                           (double4*)a, (const double4*)b, n4, (double)S);
        HIP_CHECK(hipGetLastError());
    }
    if (tail > 0) {
        hipLaunchKernelGGL(vpvts_tail_kernel, dim3(1), dim3(32), 0, 0,
                           a, b, n4 * 4, tail, (double)S);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipDeviceSynchronize());
}
