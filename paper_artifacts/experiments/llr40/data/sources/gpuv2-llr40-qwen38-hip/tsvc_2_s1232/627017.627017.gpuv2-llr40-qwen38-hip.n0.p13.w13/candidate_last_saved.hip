// Device kernels for tsvc_2_s1232.
//   for j in [0,N): for i in [j*V, N):  aa[i*N + j] = bb[i*N + j] + cc[i*N + j]
// Equivalent: for each row i, columns j = 0 .. (i/V) (capped at N-1) are updated.
// Mapping: warp spans consecutive j (contiguous in memory -> coalesced); row i per blockIdx.y.
#include <hip/hip_runtime.h>
#include <stdint.h>

__global__ void s1232_kern(double *__restrict__ aa, const double *__restrict__ bb,
                           const double *__restrict__ cc, const int64_t N, const int64_t V) {
    const int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t jV = j * V;
    for (int64_t i = blockIdx.y; i < N; i += gridDim.y) {
        if (jV > i) continue;
        const int64_t idx = i * N + j;
        aa[idx] = bb[idx] + cc[idx];
    }
}

namespace s1232 {
void launch(double *aa, const double *bb, const double *cc, int64_t N, int64_t V) {
    if (N <= 0) return;
    int64_t jmax = (V <= 0) ? N : (N - 1) / V + 1;
    if (jmax > N) jmax = N;
    if (jmax <= 0) return;
    const unsigned BX = 256;
    dim3 block(BX);
    int64_t gy = N < 65535 ? N : 65535;
    dim3 grid((unsigned)((jmax + BX - 1) / BX), (unsigned)gy);
    s1232_kern<<<grid, block>>>(aa, bb, cc, N, V);
}
} // namespace s1232
