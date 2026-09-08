// tsvc_2_s119 (s119_d_single): aa[i][j] = aa[i-1][j-1] + bb[i][j], i,j in 1..N-1.
// Dependences run only along anti-diagonals (i-j = const): each of the 2N-3
// diagonals is an independent prefix scan seeded by row 0 / column 0, which
// the kernel never writes. One warp handles one diagonal: lanes interleave
// positions p = lane, lane+W, ...; phase 1 sums each lane's chunk, a warp
// scan of the partial sums gives per-lane offsets, phase 2 re-loads bb,
// accumulates from the seed and stores aa. Every output element is one
// addition of the already-final previous value on its diagonal, so results
// are within fp reassociation tolerance of the sequential reference and are
// bitwise identical run to run.

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define HIPCHK(x)                                                                  \
  do {                                                                             \
    hipError_t e_ = (x);                                                           \
    if (e_ != hipSuccess) {                                                        \
      std::printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__,      \
                  __LINE__);                                                       \
      std::abort();                                                                \
    }                                                                              \
  } while (0)

namespace {

constexpr int BLOCK = 256;
constexpr int64_t MAX_BLOCKS = 8192;

__global__ void __launch_bounds__(BLOCK)
tsvc2_s119_scan(double* __restrict__ aa, const double* __restrict__ bb,
                const int64_t N, const int64_t nDiag, const int W) {
  const int64_t totalWarps = (int64_t)gridDim.x * (blockDim.x / W);
  const int64_t warpId = (int64_t)blockIdx.x * (blockDim.x / W) + (threadIdx.x / W);
  const int lane = threadIdx.x % W;
  const int64_t half = N - 2;  // maps di in [0, 2N-3] to dd = di - half in [-(N-2), N-2]
  const int64_t st = N + 1;    // element stride along a diagonal in flat row-major order

  for (int64_t di = warpId; di < nDiag; di += totalWarps) {
    const int64_t dd = di - half;
    const int64_t ad = dd < 0 ? -dd : dd;
    const int64_t L = N - 1 - ad;
    // first updated cell of the diagonal and the seed read before it
    const int64_t i0 = (dd >= 0) ? (dd + 1) : 1;
    const int64_t j0 = (dd >= 0) ? 1 : (-dd + 1);
    const int64_t base = i0 * N + j0;
    const int64_t seedIdx = (dd >= 0) ? (dd * N) : (-dd);  // aa[dd][0] or aa[0][-dd]

    // contiguous chunk of length c per lane: lane l covers p in [l*c, min(L,(l+1)*c))
    const int c = (int)((L + W - 1) / W);
    const int64_t p0 = (int64_t)lane * c;
    const int64_t p1 = (p0 + c < L) ? (p0 + c) : L;
    double part = 0.0;
    for (int64_t p = p0; p < p1; ++p) part += bb[base + p * st];
    double incl = part;
    for (int off = 1; off < W; off <<= 1) {
      const double u = __shfl_up(incl, off, W);
      incl = (lane >= off) ? (incl + u) : incl;
    }
    // lane l needs the sum of chunks of lanes < l; keep the shuffled register
    // read-only (in-place predicated updates before a shuffle miscompile here)
    const double excl = __shfl_up(incl, 1, W);
    double acc = aa[seedIdx] + ((lane == 0) ? 0.0 : excl);
    for (int64_t p = p0; p < p1; ++p) {
      acc += bb[base + p * st];
      aa[base + p * st] = acc;
    }
  }
}

}  // namespace

extern "C" void tsvc_2_s119_fp64(double* __restrict__ aa, const double* __restrict__ bb,
                                 const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N < 2) return;
  static int waveSize = 0;
  if (waveSize == 0) {
    hipDeviceProp_t prop;
    HIPCHK(hipGetDeviceProperties(&prop, 0));
    waveSize = prop.warpSize;
  }
  const int W = waveSize;
  const int64_t nDiag = 2 * N - 3;
  const int warpsPerBlock = BLOCK / W;
  int64_t blocks = (nDiag + warpsPerBlock - 1) / warpsPerBlock;
  if (blocks > MAX_BLOCKS) blocks = MAX_BLOCKS;
  tsvc2_s119_scan<<<(int)blocks, BLOCK, 0, hipStreamDefault>>>(aa, bb, N, nDiag, W);
  hipError_t e1 = hipGetLastError();
  if (e1 != hipSuccess) {
    std::printf("HIP error %s after launch\n", hipGetErrorString(e1));
    std::abort();
  }
}
