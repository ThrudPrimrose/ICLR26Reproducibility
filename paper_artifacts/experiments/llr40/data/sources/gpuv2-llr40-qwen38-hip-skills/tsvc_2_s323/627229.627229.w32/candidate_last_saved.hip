// TSVC tsvc_2 s323 (fp64): serial recurrence
//   a[i] = b[i-1] + c[i]*d[i];  b[i] = a[i] + c[i]*e[i]
// is a prefix sum:  b[i] = b[0] + S[i],  S[i] = sum_{j=1..i} f[j], f[j] = c[j]*(d[j]+e[j])
//   a[i] = b[0] + S[i-1] + c[i]*d[i]
// Deterministic 3-kernel block-scan (no fp64 atomics) -> bitwise-stable run to run.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define HIPCHK(x)                                                            \
  do {                                                                       \
    hipError_t e_ = (x);                                                     \
    if (e_ != hipSuccess) {                                                  \
      std::fprintf(stderr, "HIP error %s (%d) at %s:%d\n",                   \
                   hipGetErrorString(e_), (int)e_, __FILE__, __LINE__);      \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

namespace {
constexpr int T = 256;           // threads per block
constexpr int LW = 8 + 1;        // LDS slots (T/32 warp sums + total)

extern __shared__ double sds[];

// In-block exclusive scan of v (thread threadIdx.x); returns exclusive value
// for this thread and the total block sum. Fixed tree order => deterministic.
__device__ __forceinline__ double block_excl_scan(double v, double* s, double& bs_out) {
  const int W = warpSize;
  const int NW = T / W;
  const int tid = threadIdx.x;
  const int lane = tid & (W - 1);
  const int wid = tid / W;
  double inc = v, exc = 0.0;
  for (int o = 1; o < W; o <<= 1) {
    const double u = __shfl_up(inc, o, W);
    const double uc = __shfl_up(exc, o, W);
    if (lane >= o) {
      inc += u;
      exc += uc;
    }
  }
  if (lane == W - 1) s[wid] = inc;
  __syncthreads();
  if (wid == 0) {
    double xi = (lane < NW) ? s[lane] : 0.0;
    double xe = 0.0;
    for (int o = 1; o < W; o <<= 1) {
      const double u = __shfl_up(xi, o, W);
      const double ue = __shfl_up(xe, o, W);
      if (lane >= o) {
        xi += u;
        xe += ue;
      }
    }
    if (lane < NW) s[lane] = xe;  // exclusive prefix of warp totals
    if (lane == NW) s[NW] = xi;   // xi at any lane >= NW is the block total
  }
  __syncthreads();
  bs_out = s[NW];
  return (wid > 0) ? exc + s[wid - 1] : exc;
}

__global__ void __launch_bounds__(T) k_scan(const double* __restrict__ c,
                                            const double* __restrict__ d,
                                            const double* __restrict__ e,
                                            double* __restrict__ g,
                                            double* __restrict__ partials,
                                            const int64_t N) {
  const int64_t i = (int64_t)blockIdx.x * T + threadIdx.x;
  double v = (i > 0 && i < N) ? c[i] * (d[i] + e[i]) : 0.0;
  double bs;
  const double excl = block_excl_scan(v, sds, bs);
  if (i < N) g[i] = excl;
  if (threadIdx.x == 0) partials[blockIdx.x] = bs;
}

// Serial-in-order chunked exclusive scan of n partials, one block.
__global__ void __launch_bounds__(T) k_scan_partials(double* __restrict__ p, const int n) {
  const int tid = threadIdx.x;
  double acc = 0.0;
  for (int base = 0; base < n; base += T) {
    __syncthreads();  // all threads done reading s[] from previous chunk
    const int i = base + tid;
    double v = (i < n) ? p[i] : 0.0;
    double bs;
    const double excl = block_excl_scan(v, sds, bs);
    if (i < n) p[i] = acc + excl;
    acc += bs;
  }
}

__global__ void __launch_bounds__(T) k_combine(const double* __restrict__ c,
                                               const double* __restrict__ d,
                                               const double* __restrict__ e,
                                               const double* __restrict__ g,
                                               const double* __restrict__ ppart,
                                               double* __restrict__ a,
                                               double* __restrict__ b,
                                               const int64_t N) {
  const int64_t i = (int64_t)blockIdx.x * T + threadIdx.x;
  if (i < N) {
    const double b0 = b[0];
    const double excl = g[i] + ppart[blockIdx.x];
    const double ai = b0 + excl + c[i] * d[i];
    a[i] = ai;
    b[i] = ai + c[i] * e[i];
  }
}
}  // namespace

extern "C" void tsvc_2_s323_fp64(double* __restrict__ a, double* __restrict__ b,
                                 const double* __restrict__ c, const double* __restrict__ d,
                                 const double* __restrict__ e, const int64_t LEN_1D) {
  const int64_t N = LEN_1D;
  if (N < 2) return;
  static double* g = nullptr;
  static double* ppart = nullptr;
  static int64_t capN = 0;
  if (N > capN) {
    if (g) {
      HIPCHK(hipFree(g));
      HIPCHK(hipFree(ppart));
    }
    const int64_t nseg64 = (N + T - 1) / T;
    HIPCHK(hipMalloc((void**)&g, (size_t)N * sizeof(double)));
    HIPCHK(hipMalloc((void**)&ppart, (size_t)nseg64 * sizeof(double)));
    capN = N;
  }
  const int nseg = (int)((N + T - 1) / T);
  const size_t lds = (size_t)LW * sizeof(double);
  k_scan<<<nseg, T, lds>>>(c, d, e, g, ppart, N);
  HIPCHK(hipGetLastError());
  k_scan_partials<<<1, T, lds>>>(ppart, nseg);
  HIPCHK(hipGetLastError());
  k_combine<<<nseg, T>>>(c, d, e, g, ppart, a, b, N);
  HIPCHK(hipGetLastError());
  HIPCHK(hipStreamSynchronize(0));
  HIPCHK(hipGetLastError());
}
