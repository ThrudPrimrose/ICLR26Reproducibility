// TSVC tsvc_2 s2233 (fp64) -- GPU (HIP) implementation.
//
// Problem: for i in [8,N): aa[j,i] = aa[j-1,i] + cc[j,i]  (j in [8,N))
//          for i in [8,N): bb[i,j] = bb[i-1,j] + cc[i,j]  (j in [8,N))
// Both statements are, per COLUMN c in [8,N): a left-to-right accumulation
//   acc = arr[7,c]; for r in 8..N-1: acc += cc[r,c]; arr[r,c] = acc;
// independent across columns.  We run one thread per column, streaming rows
// (fully coalesced across lanes), accumulating in EXACTLY the reference order,
// so the result is bit-identical to the C reference.  cc is read once and
// feeds both accumulators.

// ============================ DEVICE HALF ============================

#include <hip/hip_runtime.h>
#include <cstdint>

__global__ void tsvc_2_s2233_scan(double* __restrict__ aa,
                                  double* __restrict__ bb,
                                  const double* __restrict__ cc,
                                  long long N) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long ncol = N - 8;
    if (t >= ncol) return;
    long long col = t + 8;
    double aacc = aa[7 * N + col];
    double bacc = bb[7 * N + col];
    const double* __restrict__ ccb = cc + col;
    double* __restrict__ aab = aa + col;
    double* __restrict__ bbb = bb + col;
    const long long stride = N;
#pragma unroll 8
    for (long long r = 8; r < N; ++r) {
        double x = ccb[r * stride];
        aacc += x;
        bacc += x;
        aab[r * stride] = aacc;
        bbb[r * stride] = bacc;
    }
}

extern "C" void tsvc_2_s2233_launch(double* aa, double* bb, const double* cc,
                                    long long N) {
    if (N <= 8) return;
    long long ncol = N - 8;
    int block = 64;
    int grid = (int)((ncol + block - 1) / block);
    tsvc_2_s2233_scan<<<grid, block>>>(aa, bb, cc, N);
}

// ============================ HOST HALF ============================

#include <cstdio>
#include <cstdlib>

#define T233_CHECK(x)                                                              \
    do {                                                                           \
        hipError_t e_ = (x);                                                       \
        if (e_ != hipSuccess) {                                                    \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_),      \
                    __FILE__, __LINE__);                                           \
            abort();                                                               \
        }                                                                          \
    } while (0)

extern "C" void tsvc_2_s2233_launch(double* aa, double* bb, const double* cc,
                                    long long N);

extern "C" void tsvc_2_s2233_fp64(double* aa, double* bb, const double* cc,
                                  long long LEN_2D,
                                  unsigned char* workspace,
                                  long long workspace_size) {
    (void)workspace;
    (void)workspace_size;
    if (LEN_2D > 8) {
        tsvc_2_s2233_launch(aa, bb, cc, LEN_2D);
        T233_CHECK(hipGetLastError());
    }
    T233_CHECK(hipDeviceSynchronize());
}
