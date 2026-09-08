// Device half: TSVC s119 on the GPU.
//
// aa[i,j] = aa[i-1,j-1] + bb[i,j]  for i,j in [1, LEN_2D).
// Each element depends ONLY on its up-left neighbour, so the dependency graph
// is a set of 2N-1 independent serial chains running along the (1,1)
// direction: one chain from every (0,c), c = 0..N-1, and one from every
// (c,0), c = 1..N-1.  Row 0 and column 0 are never written, so every chain's
// start value is read-only and every aa element is written by exactly ONE
// thread: one thread per chain is race-free and reproduces the reference
// arithmetic bit for bit (same single addition, same order).
//
// The chain walk is unrolled so eight independent bb loads are in flight at
// once: the (1,1) step only chains the carry, never the address, so a deep
// load pipeline per chain keeps the fabric full while the FADD chain runs
// behind it.  Across threads, one "phase" sweeps a contiguous band of both
// arrays, so aa and bb stream sequentially.

#include <hip/hip_runtime.h>
#include <stdint.h>

#define S119_BLOCK 256
#define S119_UNROLL 8

__global__ void s119_kernel(double * __restrict__ aa,
                            const double * __restrict__ bb,
                            int64_t N) {
  const int64_t stride = N + 1;                 // element step of the (1,1) diagonal
  const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t T = (int64_t)gridDim.x * blockDim.x;
  const int64_t nch = 2 * N - 1;                // number of chains
  for (int64_t c = tid; c < nch; c += T) {
    int64_t start, len;
    if (c < N) {                                // chain from (0,c), c = 0..N-1
      start = c;
      len = N - 1 - c;
    } else {                                    // chain from (c-N+1, 0)
      int64_t cc = c - N + 1;                   // 1..N-1
      start = cc * N;
      len = N - 1 - cc;
    }
    int64_t idx = start;
    double acc = aa[idx];
    const int64_t full = len & ~7;
    for (int64_t k = 0; k < full; k += S119_UNROLL) {
      const int64_t i0 = idx + stride;
      const int64_t i1 = i0 + stride;
      const int64_t i2 = i1 + stride;
      const int64_t i3 = i2 + stride;
      const int64_t i4 = i3 + stride;
      const int64_t i5 = i4 + stride;
      const int64_t i6 = i5 + stride;
      const int64_t i7 = i6 + stride;
      const double b0 = bb[i0];
      const double b1 = bb[i1];
      const double b2 = bb[i2];
      const double b3 = bb[i3];
      const double b4 = bb[i4];
      const double b5 = bb[i5];
      const double b6 = bb[i6];
      const double b7 = bb[i7];
      acc += b0; aa[i0] = acc;
      acc += b1; aa[i1] = acc;
      acc += b2; aa[i2] = acc;
      acc += b3; aa[i3] = acc;
      acc += b4; aa[i4] = acc;
      acc += b5; aa[i5] = acc;
      acc += b6; aa[i6] = acc;
      acc += b7; aa[i7] = acc;
      idx = i7;
    }
    for (int64_t k = full; k < len; ++k) {
      idx += stride;
      acc += bb[idx];
      aa[idx] = acc;
    }
  }
}

extern "C" void s119_launch(double *aa, const double *bb, int64_t N) {
  if (N < 2) return;
  const int64_t nch = 2 * N - 1;
  const int64_t blocks = (nch + S119_BLOCK - 1) / S119_BLOCK;
  s119_kernel<<<(int)blocks, S119_BLOCK>>>(aa, bb, N);
}
