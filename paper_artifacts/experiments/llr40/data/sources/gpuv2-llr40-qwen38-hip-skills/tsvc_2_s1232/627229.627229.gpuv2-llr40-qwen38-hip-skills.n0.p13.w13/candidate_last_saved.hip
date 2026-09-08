#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(expr) \
  do { \
    hipError_t e_ = (expr); \
    if (e_ != hipSuccess) { \
      fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
      abort(); \
    } \
  } while (0)

// ------------------------- device half -------------------------
// aa[i,j] = bb[i,j] + cc[i,j]   for 0<=j<N, j*VLEN <= i < N   (row-major, N x N)

#define TB_X 256
#define TB_Y 4

__global__ void __launch_bounds__(TB_X * TB_Y)
s1232_kernel(double* __restrict__ aa, const double* __restrict__ bb, const double* __restrict__ cc,
             int64_t n, int64_t vlen)
{
  int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // column (contiguous)
  int64_t i = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;  // row
  if (i >= n || j >= n) return;
  if (i >= j * vlen) {
    int64_t idx = i * n + j;
    aa[idx] = bb[idx] + cc[idx];
  }
}

// 1D grid-stride fallback for pathologically large N
__global__ void __launch_bounds__(256)
s1232_kernel_1d(double* __restrict__ aa, const double* __restrict__ bb, const double* __restrict__ cc,
                int64_t n, int64_t vlen)
{
  int64_t total = n * n;
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += (int64_t)gridDim.x * blockDim.x) {
    int64_t j = idx % n;
    int64_t i = idx / n;
    if (i >= j * vlen) aa[idx] = bb[idx] + cc[idx];
  }
}

extern "C" void tsvc_2_s1232_launch(double* aa, const double* bb, const double* cc,
                                    int64_t n, int64_t vlen, hipStream_t stream)
{
  if (n <= 0) return;
  if (n <= 65536LL * TB_Y) {
    dim3 block(TB_X, TB_Y);
    dim3 grid((unsigned)((n + TB_X - 1) / TB_X), (unsigned)((n + TB_Y - 1) / TB_Y));
    s1232_kernel<<<grid, block, 0, stream>>>(aa, bb, cc, n, vlen);
  } else {
    int64_t total = n * n;
    int64_t want = (total + 255) / 256;
    unsigned blocks = (unsigned)(want > 0x7FFFFFFFLL ? 0x7FFFFFFF : want);
    s1232_kernel_1d<<<blocks, 256, 0, stream>>>(aa, bb, cc, n, vlen);
  }
}

// ------------------------- host half -------------------------
extern "C" void tsvc_2_s1232_fp64(double* aa, const double* bb, const double* cc,
                                  int64_t LEN_2D, int64_t VLEN)
{
  tsvc_2_s1232_launch(aa, bb, cc, LEN_2D, VLEN, 0);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
}

#ifdef S1232_TEST_MAIN
// ------------------------- test main -------------------------
#include <vector>
#include <random>
#include <cstring>

static void host_ref(double* aa, const double* bb, const double* cc, int64_t n, int64_t vlen)
{
  for (int64_t j = 0; j < n; ++j)
    for (int64_t i = j * vlen; i < n; ++i)
      aa[i * n + j] = bb[i * n + j] + cc[i * n + j];
}

int main(int argc, char** argv)
{
  int64_t n = argc > 1 ? atoll(argv[1]) : 1001;
  int64_t vlen = argc > 2 ? atoll(argv[2]) : 1;
  int iters = argc > 3 ? atoi(argv[3]) : 5;

  std::vector<double> h_a(n * n), h_b(n * n), h_c(n * n);
  std::mt19937_64 rng(1234);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (size_t k = 0; k < h_b.size(); ++k) h_b[k] = dist(rng);
  for (size_t k = 0; k < h_c.size(); ++k) h_c[k] = dist(rng);

  double *d_a, *d_b, *d_c;
  HIP_CHECK(hipMalloc(&d_a, sizeof(double) * n * n));
  HIP_CHECK(hipMalloc(&d_b, sizeof(double) * n * n));
  HIP_CHECK(hipMalloc(&d_c, sizeof(double) * n * n));
  HIP_CHECK(hipMemcpy(d_b, h_b.data(), sizeof(double) * n * n, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_c, h_c.data(), sizeof(double) * n * n, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_a, 0xa5, sizeof(double) * n * n));  // poison

  // warmup
  tsvc_2_s1232_fp64(d_a, d_b, d_c, n, vlen);
  HIP_CHECK(hipMemcpy(h_a.data(), d_a, sizeof(double) * n * n, hipMemcpyDeviceToHost));

  hipEvent_t e0, e1;
  hipEventCreate(&e0); hipEventCreate(&e1);
  hipEventRecord(e0);
  for (int it = 0; it < iters; ++it) tsvc_2_s1232_fp64(d_a, d_b, d_c, n, vlen);
  hipEventRecord(e1);
  hipEventSynchronize(e1);
  float ms = 0.f;
  hipEventElapsedTime(&ms, e0, e1);
  double bytes = 24.0 * (double)n * n * 0.5;  // ~half of elements valid for vlen=1
  printf("n=%lld vlen=%lld: %.3f ms/iter, ~%.2f GB/s (valid-region)\n",
         (long long)n, (long long)vlen, ms / iters, bytes / (ms / iters / 1e3) / 1e9);

  // correctness vs host reference
  std::vector<double> h_ref(n * n, 1e300);
  // keep poison markers for invalid entries
  for (size_t k = 0; k < n * n; ++k) h_ref[k] = h_a[k];  // initialize with device result
  host_ref(h_ref.data(), h_b.data(), h_c.data(), n, vlen);

  // independent check: recompute and bit-compare valid region; check poison elsewhere
  int bad = 0;
  for (int64_t j = 0; j < n; ++j)
    for (int64_t i = 0; i < n; ++i) {
      size_t k = (size_t)(i * n + j);
      if (i >= j * vlen) {
        if (h_a[k] != h_ref[k]) { if (bad < 5) printf("MISMATCH at (%lld,%lld): %g vs %g\n", (long long)i, (long long)j, h_a[k], h_ref[k]); bad++; }
      } else {
        // must still hold poison pattern (not written)
        uint64_t v; std::memcpy(&v, &h_a[k], 8);
        if (v != 0xa5a5a5a5a5a5a5a5ULL) { if (bad < 5) printf("POISON LOST at (%lld,%lld)\n", (long long)i, (long long)j); bad++; }
      }
    }
  printf(bad ? "FAILED (%d bad)\n" : "PASS\n", bad);
  HIP_CHECK(hipFree(d_a)); HIP_CHECK(hipFree(d_b)); HIP_CHECK(hipFree(d_c));
  return bad ? 1 : 0;
}
#endif
