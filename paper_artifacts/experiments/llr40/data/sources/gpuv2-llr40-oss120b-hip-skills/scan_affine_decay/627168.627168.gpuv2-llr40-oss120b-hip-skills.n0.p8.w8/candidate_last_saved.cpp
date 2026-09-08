#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Definitions to allow compilation without HIP headers in syntax_check
#ifndef __HIPCC__
#define __global__
#define __host__
#define __device__
#endif

// Macro to stub hipLaunchKernelGGL for non-HIP compilation
#ifndef __HIPCC__
#define hipLaunchKernelGGL(FUNC, GRID, BLOCK, SHMEM, STREAM, ...) ((void)0)
#endif

// Minimal forward declarations for HIP runtime functions and types
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

// Transform struct matching device version
struct Transform {
    double a;
    double b;
};

struct TransformOp {
    __host__ __device__ Transform operator()(const Transform& lhs, const Transform& rhs) const {
        Transform res;
        // rhs applied after lhs (rhs ∘ lhs)
        res.a = rhs.a * lhs.a;
        res.b = rhs.a * lhs.b + rhs.b;
        return res;
    }
};

// Forward declarations of device kernels (implemented in device_source)
extern "C" __global__ void make_transform(const double* __restrict__ c,
                                          const double* __restrict__ x,
                                          Transform* __restrict__ t,
                                          int64_t N);
extern "C" __global__ void block_scan(const Transform* __restrict__ in,
                                      Transform* __restrict__ out,
                                      Transform* __restrict__ block_totals,
                                      int64_t N);
extern "C" __global__ void apply_block_prefix(const Transform* __restrict__ block_prefix,
                                              Transform* __restrict__ data,
                                              int64_t N);
extern "C" __global__ void apply_transform(const Transform* __restrict__ t,
                                           double* __restrict__ y,
                                           double y0,
                                           int64_t N);

extern "C" void scan_affine_decay_fp64(double* __restrict__ y,
                                       const double* __restrict__ c,
                                       const double* __restrict__ x,
                                       const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;
    const int threads_per_block = 256;
    int blocks = static_cast<int>((LEN_1D + threads_per_block - 1) / threads_per_block);
    if (blocks < 1) blocks = 1;

    // Allocate device buffers
    Transform *d_trans = nullptr, *d_out = nullptr, *d_block_totals = nullptr, *d_block_prefix = nullptr;
    hipError_t err;

    err = hipMalloc(reinterpret_cast<void**>(&d_trans), static_cast<size_t>(LEN_1D) * sizeof(Transform));
    if (err != hipSuccess) { std::fprintf(stderr, "hipMalloc d_trans failed: %s\n", hipGetErrorString(err)); std::abort(); }
    err = hipMalloc(reinterpret_cast<void**>(&d_out), static_cast<size_t>(LEN_1D) * sizeof(Transform));
    if (err != hipSuccess) { std::fprintf(stderr, "hipMalloc d_out failed: %s\n", hipGetErrorString(err)); std::abort(); }
    err = hipMalloc(reinterpret_cast<void**>(&d_block_totals), static_cast<size_t>(blocks) * sizeof(Transform));
    if (err != hipSuccess) { std::fprintf(stderr, "hipMalloc d_block_totals failed: %s\n", hipGetErrorString(err)); std::abort(); }

    // Step 1: create per-element transforms
    hipLaunchKernelGGL(make_transform,
                       dim3(blocks), dim3(threads_per_block), 0, 0,
                       c, x, d_trans, LEN_1D);
    err = hipGetLastError();
    if (err != hipSuccess) { std::fprintf(stderr, "make_transform launch error: %s\n", hipGetErrorString(err)); std::abort(); }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) { std::fprintf(stderr, "make_transform sync error: %s\n", hipGetErrorString(err)); std::abort(); }

    // Step 2: per-block inclusive scan and capture block totals
    hipLaunchKernelGGL(block_scan,
                       dim3(blocks), dim3(threads_per_block), 0, 0,
                       d_trans, d_out, d_block_totals, LEN_1D);
    err = hipGetLastError();
    if (err != hipSuccess) { std::fprintf(stderr, "block_scan launch error: %s\n", hipGetErrorString(err)); std::abort(); }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) { std::fprintf(stderr, "block_scan sync error: %s\n", hipGetErrorString(err)); std::abort(); }

    // Step 3: copy block totals to host and compute block prefix
    Transform* h_block_totals = (Transform*)std::malloc(static_cast<size_t>(blocks) * sizeof(Transform));
    Transform* h_block_prefix = (Transform*)std::malloc(static_cast<size_t>(blocks) * sizeof(Transform));
    if (!h_block_totals || !h_block_prefix) { std::fprintf(stderr, "malloc failed\n"); std::abort(); }
    err = hipMemcpy(h_block_totals, d_block_totals, static_cast<size_t>(blocks) * sizeof(Transform), hipMemcpyDeviceToHost);
    if (err != hipSuccess) { std::fprintf(stderr, "hipMemcpy block totals failed: %s\n", hipGetErrorString(err)); std::abort(); }

    // Compute prefix of block totals on host
    Transform identity = {1.0, 0.0};
    h_block_prefix[0] = identity;
    TransformOp op;
    for (int i = 1; i < blocks; ++i) {
        h_block_prefix[i] = op(h_block_prefix[i-1], h_block_totals[i-1]);
    }

    // Transfer block prefix to device
    err = hipMalloc(reinterpret_cast<void**>(&d_block_prefix), static_cast<size_t>(blocks) * sizeof(Transform));
    if (err != hipSuccess) { std::fprintf(stderr, "hipMalloc d_block_prefix failed: %s\n", hipGetErrorString(err)); std::abort(); }
    err = hipMemcpy(d_block_prefix, h_block_prefix, static_cast<size_t>(blocks) * sizeof(Transform), hipMemcpyHostToDevice);
    if (err != hipSuccess) { std::fprintf(stderr, "hipMemcpy block prefix failed: %s\n", hipGetErrorString(err)); std::abort(); }

    // Step 4: combine block prefix with per-block scans (skip block 0)
    hipLaunchKernelGGL(apply_block_prefix,
                       dim3(blocks), dim3(threads_per_block), 0, 0,
                       d_block_prefix, d_out, LEN_1D);
    err = hipGetLastError();
    if (err != hipSuccess) { std::fprintf(stderr, "apply_block_prefix launch error: %s\n", hipGetErrorString(err)); std::abort(); }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) { std::fprintf(stderr, "apply_block_prefix sync error: %s\n", hipGetErrorString(err)); std::abort(); }

    // Step 5: copy seed y0 = x[0] from device to host
    double y0 = 0.0;
    err = hipMemcpy(&y0, x, sizeof(double), hipMemcpyDeviceToHost);
    if (err != hipSuccess) { std::fprintf(stderr, "hipMemcpy y0 failed: %s\n", hipGetErrorString(err)); std::abort(); }

    // Step 6: apply final transform to produce output y
    hipLaunchKernelGGL(apply_transform,
                       dim3(blocks), dim3(threads_per_block), 0, 0,
                       d_out, y, y0, LEN_1D);
    err = hipGetLastError();
    if (err != hipSuccess) { std::fprintf(stderr, "apply_transform launch error: %s\n", hipGetErrorString(err)); std::abort(); }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) { std::fprintf(stderr, "apply_transform sync error: %s\n", hipGetErrorString(err)); std::abort(); }

    // Cleanup
    hipFree(d_trans);
    hipFree(d_out);
    hipFree(d_block_totals);
    hipFree(d_block_prefix);
    std::free(h_block_totals);
    std::free(h_block_prefix);
}
