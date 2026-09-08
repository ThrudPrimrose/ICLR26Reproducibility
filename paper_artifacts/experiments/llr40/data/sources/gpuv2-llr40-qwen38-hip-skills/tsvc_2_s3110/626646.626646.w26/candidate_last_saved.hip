// TSVC tsvc_2 s3110 (fp64): global max of aa (LEN_2D x LEN_2D row-major) + first
// row-major position; bb[0] = maxv + xindex + yindex.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    const char* s_ = hipGetErrorString(e_); \
    fprintf(stderr, "HIP error %s at %s:%d\n", s_, __FILE__, __LINE__); abort(); } } while (0)

// pair: (value, index); larger value wins, tie -> smaller index.
// identity: (v=-inf, i=INT64_MAX) is beaten by any real element, incl. -inf value
// (via the index tiebreak) and never beaten by NaN (NaN comparisons are false).
struct __align__(16) Pair { double v; int64_t i; };

__device__ __forceinline__ Pair better(Pair a, Pair b) {
    if (b.v > a.v || (b.v == a.v && b.i < a.i)) return b;
    return a;
}

__device__ __forceinline__ Pair warpReduce(Pair p) {
    for (int off = 32; off > 0; off >>= 1) {
        double ov = __shfl_down(p.v, off);
        int64_t oi = __shfl_down(p.i, off);
        p = better(p, Pair{ov, oi});
    }
    return p;
}

// block of 256 threads (4x64 wavefronts) -> broadcast winner
__device__ __forceinline__ Pair blockReduce(Pair p, Pair* lds) {
    int tid = threadIdx.x;
    p = warpReduce(p);
    if ((tid & 63) == 0) lds[tid >> 6] = p;
    __syncthreads();
    if (tid < 64) {
        p = (tid < 4) ? lds[tid] : Pair{-INFINITY, INT64_MAX};
        p = warpReduce(p);
        if (tid == 0) lds[0] = p;
    }
    __syncthreads();
    return lds[0];
}

extern "C" __launch_bounds__(256)
__global__ void k_main(const double4* __restrict__ aa4, const double* __restrict__ aa,
                       int64_t n4, int64_t total, Pair* __restrict__ out) {
    Pair p{-INFINITY, INT64_MAX};
    int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t st = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = g; i < n4; i += st) {
        double4 d = aa4[i];
        int64_t base = i * 4;
        p = better(p, Pair{d.x, base});
        p = better(p, Pair{d.y, base + 1});
        p = better(p, Pair{d.z, base + 2});
        p = better(p, Pair{d.w, base + 3});
    }
    for (int64_t i = n4 * 4 + g; i < total; i += st) {
        double v = aa[i];
        p = better(p, Pair{v, i});
    }
    __shared__ Pair lds[4];
    Pair r = blockReduce(p, lds);
    if (threadIdx.x == 0) out[blockIdx.x] = r;
}

extern "C" __launch_bounds__(256)
__global__ void k_final(const Pair* __restrict__ tmp, int nblocks,
                        const double* __restrict__ aa, int64_t N,
                        double* __restrict__ bb) {
    Pair p{-INFINITY, INT64_MAX};
    for (int i = threadIdx.x; i < nblocks; i += blockDim.x) p = better(p, tmp[i]);
    __shared__ Pair lds[4];
    p = blockReduce(p, lds);
    if (threadIdx.x == 0) {
        double m = p.v;
        double a0 = aa[0];
        if (a0 != a0) {  // aa[0,0] is NaN: reference maxv stays NaN forever
            bb[0] = a0;
        } else if (m == -INFINITY) {
            // no element beat -inf (all-NaN / NaN-poisoned): reference keeps aa[0,0]
            bb[0] = aa[0];
        } else {
            int64_t pos = p.i;
            int64_t x = pos / N, y = pos - x * N;
            bb[0] = m + (double)x + (double)y;
        }
    }
}

static Pair* g_tmp = nullptr;
static int g_tmp_cap = 0;

extern "C" void tsvc_2_s3110_fp64(const double* __restrict__ aa, double* __restrict__ bb,
                                  int64_t LEN_2D) {
    int64_t total = LEN_2D * LEN_2D;
    int64_t n4 = total / 4;

    int cus = 304;
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    if (prop.multiProcessorCount > 0) cus = prop.multiProcessorCount;

    int gblocks = (int)((n4 + 1023) / 1024); // 1 double4 per thread of 8 blocks/CU
    int maxb = cus * 8;
    if (gblocks > maxb) gblocks = maxb;
    if (gblocks < 1) gblocks = 1;

    if (gblocks > g_tmp_cap) {
        if (g_tmp) HIP_CHECK(hipFree(g_tmp));
        g_tmp = nullptr;
        g_tmp_cap = 0;
        HIP_CHECK(hipMalloc((void**)&g_tmp, sizeof(Pair) * gblocks));
        g_tmp_cap = gblocks;
    }

    k_main<<<gblocks, 256>>>(reinterpret_cast<const double4*>(aa), aa, n4, total, g_tmp);
    HIP_CHECK(hipGetLastError());
    k_final<<<1, 256>>>(g_tmp, gblocks, aa, LEN_2D, bb);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(0));
}
