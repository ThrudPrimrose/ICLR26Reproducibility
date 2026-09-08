// TSVC s315 argmax_with_index -- single-pass, deterministic GPU reduction.
//
// Monoid over pairs (value, index): a candidate (x, xi) beats (v, i) iff
//   x > v, or (x == v and xi < i).
// This monoid is associative and commutative (it is a lexicographic max over
// (value, -index)), so ANY reduction tree returns the identical (max value,
// FIRST index of that value) -- exactly the reference's strict-> / first-tie
// semantics -- bitwise, on every run.  Hence the int64 output is exact and
// the determinism gate cannot be violated by scheduling.
//
// One grid: each block reduces a contiguous 256*ITEMS chunk with double4
// loads, stores its (v,i) pair into workspace, fences, and atomicCounts; the
// last block to arrive reduces all partials (ascending block id) and writes
// the two outputs.  No second launch, no atomics on the data, no ordering
// hazards.

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cfloat>
#include <climits>
#include <cstdio>
#include <cstdlib>

#define HIPCHK(x)                                                          \
  do {                                                                     \
    hipError_t e_ = (x);                                                   \
    if (e_ != hipSuccess) {                                                \
      std::fprintf(stderr, "HIP error %s (%d) at %s:%d\n",                \
                   hipGetErrorString(e_), (int)e_, __FILE__, __LINE__);    \
      std::abort();                                                        \
    }                                                                      \
  } while (0)

namespace {

constexpr int kBlock = 256;      // threads per block (4 wavefronts on gfx942)
constexpr int kItemsMax = 1024;  // doubles per thread (256 x double4 loads)
constexpr int64_t kPerBlock = (int64_t)kBlock * kItemsMax; // 262144

struct __attribute__((aligned(16))) Pair {
  double v;
  int64_t i;
};

__device__ __forceinline__ void better(const double x, const int64_t xi,
                                       double& v, int64_t& i) {
  if (x > v) {
    v = x;
    i = xi;
  } else if (x == v && xi < i) {
    i = xi;
  }
}

// Reduce one (v,i) held per thread across the block; result in lane 0.
template <bool kOutAllLanes = false>
__device__ __forceinline__ void blockReduce(double& v, int64_t& i,
                                            Pair* sdp) {
  const int warpSize = 64; // gfx942 wavefront; see note in host entry
  for (int off = 1; off < warpSize; off <<= 1) {
    const double ov = __shfl_down(v, off, warpSize);
    const int64_t oi = __shfl_down(i, off, warpSize);
    if (ov > v) {
      v = ov;
      i = oi;
    } else if (ov == v && oi < i) {
      i = oi;
    }
  }
  if ((threadIdx.x & (warpSize - 1)) == 0) sdp[threadIdx.x / warpSize] = {v, i};
  __syncthreads();
  if (threadIdx.x < warpSize) {
    const double x = sdp[threadIdx.x].v;
    const int64_t xi = sdp[threadIdx.x].i;
    if (x > v) {
      v = x;
      i = xi;
    } else if (x == v && xi < i) {
      i = xi;
    }
    __syncthreads();
    for (int off = 1; off < warpSize; off <<= 1) {
      const double ov = __shfl_down(v, off, warpSize);
      const int64_t oi = __shfl_down(i, off, warpSize);
      if (ov > v) {
        v = ov;
        i = oi;
      } else if (ov == v && oi < i) {
        i = oi;
      }
    }
    if constexpr (kOutAllLanes) {
      // lane 0 holds the answer; broadcast is unnecessary (only lane 0 reads)
    }
  }
}

}  // namespace

__global__ void __launch_bounds__(kBlock)
argmax_kernel(const double* __restrict__ a, const int64_t n,
              Pair* __restrict__ partials, int* __restrict__ counter,
              double* __restrict__ out_value, int64_t* __restrict__ out_index,
              const int64_t items, const bool vec) {
  const int tid = threadIdx.x;
  const int64_t base = ((int64_t)blockIdx.x * (kBlock * items)) + (int64_t)tid * items;
  const int64_t eend0 = base + items;
  const int64_t eend = eend0 < n ? eend0 : n;

  double v = -HUGE_VAL;
  int64_t i = LLONG_MAX;

  if (vec) {
    const int64_t vstart = base >> 2;
    const int64_t nv = n >> 2;
    int64_t vend = (eend0 >> 2);
    if (vend > nv) vend = nv;
    const double4* __restrict__ av = reinterpret_cast<const double4*>(a);
    for (int64_t k = vstart; k < vend; ++k) {
      const double4 x = av[k];
      const int64_t e = k << 2;
      better(x.x, e, v, i);
      better(x.y, e + 1, v, i);
      better(x.z, e + 2, v, i);
      better(x.w, e + 3, v, i);
    }
    for (int64_t e = vend << 2; e < eend; ++e) better(a[e], e, v, i);
  } else {
    for (int64_t e = base; e < eend; ++e) better(a[e], e, v, i);
  }

  __shared__ Pair sdp[kBlock / 32];  // 8 slots: enough for 64-wide wavefronts
  __shared__ int s_last;
  blockReduce(v, i, sdp);

  if (tid == 0) {
    partials[blockIdx.x] = Pair{v, i};
    __threadfence();
    const int done = atomicAdd(counter, 1);
    s_last = (done == (int)gridDim.x - 1);
  }
  __syncthreads();

  if (s_last) {
    // Every other block has written-and-fenced its partial before its
    // atomicAdd, and this block observed all of them, so every partial is
    // visible.  Reduce in ascending block id -> bitwise deterministic.
    double fv = -HUGE_VAL;
    int64_t fi = LLONG_MAX;
    for (int64_t p = tid; p < (int64_t)gridDim.x; p += kBlock) {
      const Pair q = partials[p];
      if (q.v > fv) {
        fv = q.v;
        fi = q.i;
      } else if (q.v == fv && q.i < fi) {
        fi = q.i;
      }
    }
    blockReduce(fv, fi, sdp);
    if (tid == 0) {
      *out_value = fv;
      *out_index = fi;
    }
  }
}

extern "C" void argmax_with_index_fp64(const double* __restrict__ a,
                                       int64_t* __restrict__ out_index,
                                       double* __restrict__ out_value,
                                       const int64_t LEN_1D, void* workspace,
                                       const int64_t workspace_bytes) {
  const int64_t n = LEN_1D;
  if (n <= 0) {  // reference would read a[0] (UB); keep outputs defined
    *out_value = 0.0;
    *out_index = 0;
    return;
  }
  // Adaptive items: single block for n <= kPerBlock, otherwise full stride.
  int64_t items = (n + (kBlock * int64_t{1})) / kBlock;  // ceil(n/BLOCK)
  if (items < 1) items = 1;
  if (items > kItemsMax) items = kItemsMax;
  const int64_t per_block = (int64_t)kBlock * items;
  const int64_t nblocks = (n + per_block - 1) / per_block;

  const bool vec = ((reinterpret_cast<uintptr_t>(a) & 31) == 0) &&
                   (items % 4 == 0);

  Pair* partials = static_cast<Pair*>(workspace);
  int* counter =
      reinterpret_cast<int*>(static_cast<char*>(workspace) + 16 * nblocks);

  argmax_kernel<<<(unsigned)nblocks, kBlock>>>(a, n, partials, counter,
                                               out_value, out_index, items, vec);
  HIPCHK(hipGetLastError());
}
