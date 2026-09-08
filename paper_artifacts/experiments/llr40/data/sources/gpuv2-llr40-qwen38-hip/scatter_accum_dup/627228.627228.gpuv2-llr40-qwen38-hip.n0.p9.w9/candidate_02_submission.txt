// Device half: scatter accumulate bins[ip[i]] += src[i]; ip may repeat an index,
// so the accumulation is a genuine reduction: use a global fp64 atomic add.
#include <hip/hip_runtime.h>
#include <cstdint>

extern "C" {

__global__ void scatter_accum_kernel(double* __restrict__ bins,
                                     const int32_t* __restrict__ ip,
                                     const double* __restrict__ src,
                                     const int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&bins[(int64_t)ip[i]], src[i]);
    }
}

void scatter_accum_dup_launch(double* bins, const int32_t* ip, const double* src, const int64_t n) {
    if (n <= 0) return;
    constexpr int kBlock = 256;
    const int64_t grid = (n + kBlock - 1) / kBlock;
    scatter_accum_kernel<<<(unsigned)grid, kBlock>>>(bins, ip, src, n);
}

}  // extern "C"
