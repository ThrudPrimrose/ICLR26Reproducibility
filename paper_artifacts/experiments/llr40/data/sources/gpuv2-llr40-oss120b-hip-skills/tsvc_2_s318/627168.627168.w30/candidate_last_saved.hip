#ifndef __HIP_PLATFORM_AMD__
// Stub definitions for syntax checking without HIP headers
#include <cstddef>
#include <cstdio>
#include <cmath>
#include <cinttypes>
typedef int hipError_t;
static const hipError_t hipSuccess = 0;
static inline hipError_t hipMalloc(void** p, std::size_t) { *p = nullptr; return hipSuccess; }
static inline hipError_t hipFree(void*) { return hipSuccess; }
static inline hipError_t hipDeviceSynchronize() { return hipSuccess; }
static inline hipError_t hipGetLastError() { return hipSuccess; }
static inline const char* hipGetErrorString(hipError_t) { return "hip stub"; }
#define hipLaunchKernelGGL(kernel, grid, block, shmem, stream, ...) ((void)0)
#else
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <cinttypes>
#endif

#ifndef __HIP_PLATFORM_AMD__
#ifndef __global__
#define __global__
#endif
#endif

#ifndef __HIP_PLATFORM_AMD__
// Stub dim3 definition for syntax checking
struct dim3 {
    unsigned int x, y, z;
    constexpr dim3(unsigned int x_=1, unsigned int y_=1, unsigned int z_=1) : x(x_), y(y_), z(z_) {}
};
#endif

#define HIP_CHECK(err) do { \
    hipError_t err_ = (err); \
    if (err_ != hipSuccess) { \
        fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err_)); \
        exit(1); \
    } \
} while (0)

struct Pair {
    double maxv;
    int64_t idx;
};

// Declarations of device kernels defined in the device source
extern "C" __global__ void compute_kernel(const double* __restrict__ a, int64_t N, int64_t inc, double* __restrict__ result);

extern "C" void tsvc_2_s318_fp64(const double* __restrict__ a,
                                 double* __restrict__ result,
                                 const int64_t inc,
                                 const int64_t LEN_1D) {
    // Launch a single-thread kernel to compute the result directly
    hipLaunchKernelGGL(compute_kernel, dim3(1), dim3(1), 0, 0,
                       a, LEN_1D, inc, result);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
}

