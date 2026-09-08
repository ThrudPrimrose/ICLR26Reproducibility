// TSVC tsvc_2 s233, fp64: two axis-1 prefix scans.
//   aa[j,i] = aa[j-1,i] + cc[j,i]  (down-column scan per column i, j from 8)
//   bb[j,i] = bb[j,i-1] + cc[j,i]  (right-row scan per row j, i from 8)
// v1: one thread per column (aa) / one thread per row (bb).
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

#define HIPCHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); fflush(stdout); abort(); } } while (0)

__global__ void scan_cols_aa(double* __restrict__ aa, const double* __restrict__ cc,
                             const int64_t N, const int64_t L) {
  int64_t col = (int64_t)blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (col >= L) return;
  int64_t ci = col + 8;  /* real column index; only cols 8..N-1 are updated */
  double s = aa[7 * N + ci];
  for (int64_t j = 8; j < N; ++j) {
    s += cc[j * N + ci];
    aa[j * N + ci] = s;
  }
}

__global__ void scan_rows_bb(double* __restrict__ bb, const double* __restrict__ cc,
                             const int64_t N, const int64_t L) {
  int64_t row = (int64_t)blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (row >= L) return;
  int64_t rj = row + 8;  /* real row index; only rows 8..N-1 are updated */
  double s = bb[rj * N + 7];
  for (int64_t i = 8; i < N; ++i) {
    s += cc[rj * N + i];
    bb[rj * N + i] = s;
  }
}

extern "C" void tsvc_2_s233_fp64(double* aa, double* bb, const double* cc, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  const int64_t L = N - 8;
  printf("tsvc_2_s233 LEN_2D = %ld\n", (long)N);
  fflush(stdout);
  if (L <= 0) return;
  const int BLOCK = 256;
  const int64_t grid = (L + BLOCK - 1) / BLOCK;
  hipLaunchKernelGGL(scan_cols_aa, dim3((unsigned)grid), dim3(BLOCK), 0, 0, aa, cc, N, L);
  HIPCHECK(hipGetLastError());
  hipLaunchKernelGGL(scan_rows_bb, dim3((unsigned)grid), dim3(BLOCK), 0, 0, bb, cc, N, L);
  HIPCHECK(hipGetLastError());
  HIPCHECK(hipDeviceSynchronize());
}
