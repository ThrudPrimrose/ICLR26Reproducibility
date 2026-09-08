// tsvc_2_s2275: aa[i] += bb[i]*cc[i] over LEN_2D^2 elements; a[i] = b[i] + c[i]*d[i].
// Pure elementwise -> one thread per element, fully bandwidth-bound.
#include <hip/hip_runtime.h>
#include <cstdint>

__global__ __launch_bounds__(256)
void tsvc_body(double* __restrict__ a, double* __restrict__ aa,
               const double* __restrict__ b, const double* __restrict__ bb,
               const double* __restrict__ c, const double* __restrict__ cc,
               const double* __restrict__ d, const long long n2, const long long len)
{
    const long long t = (long long)blockIdx.x * 256 + threadIdx.x;
    if (t < n2) aa[t] = aa[t] + bb[t] * cc[t];
    if (t < len) a[t] = b[t] + c[t] * d[t];
}

extern "C" void tsvc_2_s2275_fp64(double* a, double* aa, const double* b, const double* bb,
                                  const double* c, const double* cc, const double* d,
                                  const int64_t LEN_2D)
{
    const long long n2 = (long long)LEN_2D * LEN_2D;
    long long blocks = (n2 + 255) / 256;
    if (blocks < 1) blocks = 1;
    tsvc_body<<<(unsigned)blocks, 256>>>(a, aa, b, bb, c, cc, d, n2, (long long)LEN_2D);
}
