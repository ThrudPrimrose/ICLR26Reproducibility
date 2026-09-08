#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s (%d) at %s:%d\n", hipGetErrorString(e_), (int)e_, __FILE__, __LINE__); \
    abort(); } } while(0)

__global__ void __launch_bounds__(256) s1244_main(const double * __restrict__ a_in,
                                                   const double * __restrict__ b,
                                                   const double * __restrict__ c,
                                                   double * __restrict__ a,
                                                   double * __restrict__ d,
                                                   long n) {
    long i = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;
    if (i < n) {
        double cc = c[i];
        double bb = b[i];
        double ai = bb + cc*cc + bb*bb + cc;
        a[i] = ai;
        d[i] = ai + a_in[i+1];
    }
}

extern "C" void tsvc_2_s1244_fp64(double *a, const double *b, const double *c, double *d, long long LEN_1D) {
    long n = (long)LEN_1D;
    long total = n - 1;
    if (total <= 0) return;

    static double *a_in = nullptr;
    static std::size_t cap = 0;
    if (cap < (std::size_t)n) {
        if (a_in) { hipError_t fe = hipFree(a_in); (void)fe; }
        HIP_CHECK(hipMalloc(&a_in, (std::size_t)n * sizeof(double)));
        cap = (std::size_t)n;
    }
    HIP_CHECK(hipMemcpy(a_in, a, (std::size_t)n * sizeof(double), hipMemcpyDeviceToDevice));

    const int block = 256;
    long grid = (total + block - 1) / block;
    s1244_main<<<(unsigned)grid, block>>>(a_in, b, c, a, d, total);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
}
