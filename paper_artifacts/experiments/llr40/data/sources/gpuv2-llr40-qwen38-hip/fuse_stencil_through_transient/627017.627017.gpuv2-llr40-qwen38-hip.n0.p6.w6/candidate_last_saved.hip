#include <hip/hip_runtime.h>
#include <stdint.h>

#define BLOCK 256

__global__ void fst_kernel(const double* __restrict__ a, double* __restrict__ out, long long n) {
    long long i = (long long)blockIdx.x * BLOCK + threadIdx.x;
    if (i < 1 || i >= n - 2) return;
    out[i] = (a[i-1] + a[i] + a[i+1]) * (a[i] + a[i+1] + a[i+2]);
}

extern "C" void fuse_stencil_through_transient_fp64(const double* __restrict__ a, double* __restrict__ out, const int64_t LEN_1D) {
    long long n = (long long)LEN_1D;
    if (n < 4) return;
    long long grid = (n + BLOCK - 1) / BLOCK;
    fst_kernel<<<(unsigned)grid, BLOCK>>>(a, out, n);
}
