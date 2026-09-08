#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef __HIPCC__
#include "/opt/rocm/include/hip/hip_runtime.h"
#else
#define __global__
#define __host__
#define __device__
struct dim3 { unsigned int x, y, z; constexpr dim3(unsigned int x_=1, unsigned int y_=1, unsigned int z_=1) : x(x_), y(y_), z(z_) {} };
#define hipLaunchKernelGGL(FUNC, GRID, BLOCK, SHMEM, STREAM, ...) ((void)0)
typedef int hipError_t;
constexpr hipError_t hipSuccess = 0;
extern "C" const char* hipGetErrorString(hipError_t);
extern "C" hipError_t hipMalloc(void** ptr, size_t size);
extern "C" hipError_t hipFree(void* ptr);
extern "C" hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, int kind);
extern "C" hipError_t hipDeviceSynchronize();
extern "C" hipError_t hipGetLastError();
constexpr int hipMemcpyDeviceToHost = 2;
constexpr int hipMemcpyHostToDevice = 1;
#endif

extern "C" __global__ void scan_seq(double* __restrict__ y,
                                    const double* __restrict__ c,
                                    const double* __restrict__ x,
                                    int64_t N);

extern "C" void scan_affine_decay_fp64(double* __restrict__ y,
                                       const double* __restrict__ c,
                                       const double* __restrict__ x,
                                       const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;
    hipLaunchKernelGGL(scan_seq,
                       dim3(1), dim3(1), 0, 0,
                       y, c, x, LEN_1D);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::fprintf(stderr, "scan_seq launch error: %s\n", hipGetErrorString(err));
        std::abort();
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::fprintf(stderr, "scan_seq sync error: %s\n", hipGetErrorString(err));
        std::abort();
    }
}
