// segment_reduce_ragged -- HIP (gfx942)
// out[s] = sum_{e in [row_ptr[s], row_ptr[s+1])} val[e]*w[e]   (float64)
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(x)                                                            \
  do {                                                                          \
    hipError_t e_ = (x);                                                        \
    if (e_ != hipSuccess) {                                                     \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_),         \
              __FILE__, __LINE__);                                              \
      abort();                                                                  \
    }                                                                           \
  } while (0)

// One thread per segment; the thread owns the whole (short) segment, so each
// element of val/w is read exactly once and no atomics are needed.
__global__ void seg_reduce(const int64_t *__restrict__ row_ptr,
                           const double *__restrict__ val,
                           const double *__restrict__ w,
                           double *__restrict__ out,
                           const int64_t nseg) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t s = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; s < nseg; s += stride) {
    const int64_t begin = row_ptr[s];
    const int64_t end = row_ptr[s + 1];
    double acc = 0.0;
    int64_t i = begin;
    for (; i + 3 < end; i += 4) {
      acc += val[i] * w[i];
      acc += val[i + 1] * w[i + 1];
      acc += val[i + 2] * w[i + 2];
      acc += val[i + 3] * w[i + 3];
    }
    for (; i < end; ++i) {
      acc += val[i] * w[i];
    }
    out[s] = acc;
  }
}

extern "C" void segment_reduce_ragged(const int64_t *row_ptr, const double *val,
                                      const double *w, double *out,
                                      int64_t nseg, double * /*ws*/,
                                      int64_t /*ws_bytes*/) {
  const int tpb = 256;
  const int64_t nblocks = (nseg + tpb - 1) / tpb;
  seg_reduce<<<(unsigned)nblocks, tpb>>>(row_ptr, val, w, out, nseg);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipGetLastError());
}
