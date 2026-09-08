// scan_affine_decay: y[i] = c[i]*y[i-1] + x[i]  (y[0] given), variable-coefficient affine scan.
// Single-kernel decoupled lookback over affine map composition:
//   map F_i(v) = c[i]*v + x[i];  y[i] = (F_i o F_{i-1} o ... o F_1)(y[0])
//   compose:  P o Q  =>  (P.A*Q.A, P.A*Q.B + P.B)
// Block b computes its local inclusive scans, publishes block total, looks back to block b-1,
// applies carry, writes y. No extra per-element traffic: 16 B read + 8 B written per element.

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

#define TPB 256
#define MAXW 8

#define HIPCHK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::printf("HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); abort(); } } while (0)

__device__ __forceinline__ void shfl_xor_dd(double &va, double &vb, unsigned mask)
{
    va = __shfl_xor(va, mask);
    vb = __shfl_xor(vb, mask);
}

extern "C" __global__ void
scan_decay_kernel(double * __restrict__ y,
                  const double * __restrict__ c,
                  const double * __restrict__ x,
                  const long long N,
                  const unsigned long long gen,
                  double * __restrict__ totA,
                  double * __restrict__ totB,
                  unsigned long long * __restrict__ flags)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const long long idx = (long long)bid * TPB + tid;

    __shared__ double sW[2 * MAXW];
    __shared__ double sCA, sCB;

    const int wsize = warpSize;
    const int lane  = tid & (wsize - 1);
    const int nwarp = TPB / wsize;
    const int warp  = tid / wsize;

    double a, b;
    if (idx < N && !(bid == 0 && tid == 0)) { a = c[idx]; b = x[idx]; }
    else { a = 1.0; b = 0.0; }

    // warp-level inclusive scan: new = F_this o F_lower
    for (int off = wsize >> 1; off > 0; off >>= 1) {
        double oa = a, ob = b;
        shfl_xor_dd(oa, ob, (unsigned)off);
        b = a * ob + b;
        a = a * oa;
    }

    // cross-warp: warp totals -> exclusive carries per warp; thread0 gets block total + publishes
    if (lane == wsize - 1) { sW[warp] = a; sW[MAXW + warp] = b; }
    __syncthreads();
    if (tid == 0) {
        double ca = 1.0, cb = 0.0;
        for (int i = 0; i < nwarp; ++i) {
            double wa = sW[i], wb = sW[MAXW + i];
            sW[i] = ca; sW[MAXW + i] = cb;            // exclusive carry for warp i
            double nca = wa * ca;
            double ncb = wa * cb + wb;
            ca = nca; cb = ncb;
        }
        totA[bid] = ca; totB[bid] = cb;
        __threadfence();
        flags[bid] = gen;
    }
    __syncthreads();

    double CA, CB;
    if (bid > 0) {
        if (tid == 0) {
            const volatile unsigned long long *vf = flags;
            while (vf[bid - 1] != gen) { }
            __threadfence();
            sCA = totA[bid - 1];
            sCB = totB[bid - 1];
        }
        __syncthreads();
        CA = sCA; CB = sCB;
    } else { CA = 1.0; CB = 0.0; }

    if (idx < N) {
        double y0 = y[0];
        y[idx] = fma(a, fma(CA, y0, CB), b);
    }
}

static double *g_totA = nullptr;
static double *g_totB = nullptr;
static unsigned long long *g_flags = nullptr;
static size_t g_cap = 0;
static unsigned long long g_gen = 0;

static void ensure_scratch(size_t G)
{
    if (g_cap >= G) return;
    size_t cap = g_cap ? g_cap : 1024;
    while (cap < G) cap <<= 1;
    if (g_totA) { HIPCHK(hipFree(g_totA)); HIPCHK(hipFree(g_totB)); HIPCHK(hipFree(g_flags)); }
    HIPCHK(hipMalloc(&g_totA, cap * sizeof(double)));
    HIPCHK(hipMalloc(&g_totB, cap * sizeof(double)));
    HIPCHK(hipMalloc(&g_flags, cap * sizeof(unsigned long long)));
    g_cap = cap;
}

extern "C" void scan_affine_decay(double *y, const double *c, const double *x, long long LEN_1D)
{
    const long long N = LEN_1D;
    if (N <= 1) return;
    const size_t G = (size_t)((N + TPB - 1) / TPB);
    ensure_scratch(G);
    ++g_gen;
    scan_decay_kernel<<<(unsigned)G, TPB, 0, (hipStream_t)0>>>(
        y, c, x, N, g_gen, g_totA, g_totB, g_flags);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) { std::printf("launch error: %s\n", hipGetErrorString(e)); }
}
