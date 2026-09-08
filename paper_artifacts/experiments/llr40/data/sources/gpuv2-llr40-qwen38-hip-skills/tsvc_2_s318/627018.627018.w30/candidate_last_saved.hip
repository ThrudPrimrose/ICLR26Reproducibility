/* tsvc_2_s318: strided argmax of fabs over a[0::inc]; result = maxv + (double)first_index.
 * Deterministic NaN-safe total order: larger |v| wins; NaN loses to every non-NaN;
 * ties -> smaller index. Order-independent => bitwise-identical across runs. */
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define TSVC_MAXG 16384
#define TSVC_BLK  256
#define TSVC_WAVE 64
#define TSVC_NW   (TSVC_BLK / TSVC_WAVE)

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s (%s:%d)\n", hipGetErrorString(e_), __FILE__, __LINE__); \
    std::abort(); } } while (0)

__device__ double   g_partV[TSVC_MAXG];
__device__ int64_t  g_partI[TSVC_MAXG];

__device__ __forceinline__ void tsvc_combine(double &bv, int64_t &bi, double pv, int64_t pi) {
    const bool bnan = std::isnan(bv), pnan = std::isnan(pv);
    const bool takep = (pv > bv)
        || (bnan && !pnan)
        || (!bnan && !pnan && pv == bv && pi < bi)
        || (bnan && pnan && pi < bi);
    if (takep) { bv = pv; bi = pi; }
}

/* block reduce -> g_partV[blockIdx.x]; all threads hit every barrier */
__device__ __forceinline__ void tsvc_block_reduce(double bv, int64_t bi) {
    __shared__ double   sV[TSVC_NW];
    __shared__ int64_t  sI[TSVC_NW];
    #pragma unroll
    for (int off = TSVC_WAVE / 2; off > 0; off >>= 1) {
        double pv = __shfl_down(bv, off, TSVC_WAVE);
        uint32_t lo = __shfl_down((uint32_t)(uint64_t)bi, off, TSVC_WAVE);
        uint32_t hi = __shfl_down((uint32_t)((uint64_t)bi >> 32), off, TSVC_WAVE);
        int64_t pi = (int64_t)lo | ((int64_t)hi << 32);
        tsvc_combine(bv, bi, pv, pi);
    }
    const int lane = threadIdx.x & (TSVC_WAVE - 1), w = threadIdx.x >> 6;
    if (lane == 0) { sV[w] = bv; sI[w] = bi; }
    __syncthreads();
    if (w == 0 && lane == 0) {
        #pragma unroll
        for (int j = 1; j < TSVC_NW; ++j) tsvc_combine(bv, bi, sV[j], sI[j]);
        if (w == 0) { g_partV[blockIdx.x] = bv; g_partI[blockIdx.x] = bi; }
    }
}

__global__ __launch_bounds__(TSVC_BLK) void tsvc_k1(const double* __restrict__ a,
                                                     const int64_t N, const int64_t inc,
                                                     const int64_t G) {
    const int64_t stride = G * (int64_t)TSVC_BLK;
    int64_t i = (int64_t)blockIdx.x * TSVC_BLK + threadIdx.x;
    double bv = -1.0; int64_t bi = -1;
    if (i < N) {
        int64_t k = i * inc, kstep = stride * inc;
        double v = __builtin_fabs(a[k]);
        int64_t ii = i;
        i += stride; k += kstep;
        while (i < N) {
            tsvc_combine(bv, bi, v, ii);
            v = __builtin_fabs(a[k]);
            ii = i;
            i += stride; k += kstep;
        }
        tsvc_combine(bv, bi, v, ii);
    }
    /* block 0 wave 0: stash element 0 raw for the NaN gate (bv is fabs already) */
    tsvc_block_reduce(bv, bi);
}

__global__ __launch_bounds__(TSVC_BLK) void tsvc_k2(const double* __restrict__ a,
                                                     double* __restrict__ result,
                                                     const int64_t G) {
    double bv = -1.0; int64_t bi = -1;
    for (int g = threadIdx.x; g < G; g += TSVC_BLK)
        tsvc_combine(bv, bi, g_partV[g], g_partI[g]);
    /* reuse block reduce into locals: copy of the same routine */
    __shared__ double   sV[TSVC_NW];
    __shared__ int64_t  sI[TSVC_NW];
    #pragma unroll
    for (int off = TSVC_WAVE / 2; off > 0; off >>= 1) {
        double pv = __shfl_down(bv, off, TSVC_WAVE);
        uint32_t lo = __shfl_down((uint32_t)(uint64_t)bi, off, TSVC_WAVE);
        uint32_t hi = __shfl_down((uint32_t)((uint64_t)bi >> 32), off, TSVC_WAVE);
        int64_t pi = (int64_t)lo | ((int64_t)hi << 32);
        tsvc_combine(bv, bi, pv, pi);
    }
    const int lane = threadIdx.x & (TSVC_WAVE - 1), w = threadIdx.x >> 6;
    if (lane == 0) { sV[w] = bv; sI[w] = bi; }
    __syncthreads();
    if (w == 0 && lane == 0) {
        #pragma unroll
        for (int j = 1; j < TSVC_NW; ++j) tsvc_combine(bv, bi, sV[j], sI[j]);
        double r = (bi < 0) ? 0.0 : bv + (double)bi;
        if (std::isnan(a[0])) r = NAN;   /* serial semantics: NaN at a[0] poisons result */
        result[0] = r;
    }
}

extern "C" void tsvc_launch(const double* a, double* result, int64_t N, int64_t inc) {
    if (N <= 0) { HIP_CHECK(hipStreamSynchronize(0)); return; }
    int64_t G = (N + TSVC_BLK - 1) / TSVC_BLK;
    if (G > TSVC_MAXG) G = TSVC_MAXG;
    if (G < 1) G = 1;
    tsvc_k1<<<(unsigned)G, TSVC_BLK>>>(a, N, inc, G);
    HIP_CHECK(hipGetLastError());
    tsvc_k2<<<1, TSVC_BLK>>>(a, result, G);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(0));
}
