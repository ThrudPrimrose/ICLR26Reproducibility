// tsvc_2_s255: a[i] = (b[i] + x + y) * 0.333 with x, y a two-step lookback over b
// (wrapping at the end). Equivalent to a[i] = ((b[i] + b[(i-1) mod N]) + b[(i-2) mod N]) * 0.333,
// which is fully parallel and bit-identical (same op order) to the sequential reference.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(x) do { hipError_t e__ = (x); if (e__ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e__), __FILE__, __LINE__); \
    abort(); } } while (0)

__global__ __launch_bounds__(256) void s255_kernel(
        double* __restrict__ a, const double* __restrict__ b, const int64_t n) {
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t npairs = (n + 1) >> 1;  // number of work units
    for (int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; t < npairs; t += stride) {
        const int64_t i = 2 * t;
        if (i + 1 < n) {
            if (t == 0) {
                // a[0], a[1]: lookback wraps around the tail of b
                const double b0 = b[0];
                const double b1 = b[1];
                a[0] = ((b0 + b[n-1]) + b[n-2]) * 0.333;
                a[1] = ((b1 + b0) + b[n-1]) * 0.333;
            } else {
                const double2 v = reinterpret_cast<const double2*>(b)[i >> 1]; // {b[i], b[i+1]}
                const double s1 = b[i-1];
                const double s2 = b[i-2];
                double2 r;
                r.x = ((v.x + s1) + s2) * 0.333;  // a[i]
                r.y = ((v.y + v.x) + s1) * 0.333; // a[i+1]
                reinterpret_cast<double2*>(a)[i >> 1] = r;
            }
        } else {
            // odd n: final element i == n-1 (n >= 3 here)
            a[i] = ((b[i] + b[i-1]) + b[i-2]) * 0.333;
        }
    }
}

extern "C" void tsvc_2_s255_fp64(double* __restrict__ a, const double* __restrict__ b,
                                 const int64_t n) {
    if (n <= 0) return;
    if (n == 1) { a[0] = ((b[0] + b[0]) + b[0]) * 0.333; return; }
    const int64_t npairs = (n + 1) >> 1;
    const int block = 256;
    int64_t grid = (npairs + block - 1) / block;
    if (grid > 1048576) grid = 1048576;
    hipLaunchKernelGGL(s255_kernel, dim3((unsigned)grid), dim3(block), 0, 0, a, b, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
}
