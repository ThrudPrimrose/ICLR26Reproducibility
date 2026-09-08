// Device half of the tsvc_2_s252 HIP submission.
//
// Reference (serial):
//   t = 0; for i in 0..n-1: s = b[i]*c[i]; a[i] = s + t; t = s;
// Since t always equals the PREVIOUS iteration's s, the recurrence is just a
// one-element shift:  a[i] = b[i]*c[i] + b[i-1]*c[i-1]  (a[0] = b[0]*c[0]).
// Every output element is therefore independent: fully parallel.
#include <hip/hip_runtime.h>
#include <cstdint>

__global__ void __launch_bounds__(256) s252_v8(double* __restrict__ a,
                                               const double* __restrict__ b,
                                               const double* __restrict__ c,
                                               long long n) {
    const long long stride = (long long)gridDim.x * blockDim.x * 8;
    long long i0 = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    for (; i0 < n; i0 += stride) {
        if (i0 + 8 <= n) {
            const double2* bb = (const double2*)(b + i0);
            const double2* cc = (const double2*)(c + i0);
            double2 b0 = bb[0], b1 = bb[1], b2 = bb[2], b3 = bb[3];
            double2 c0 = cc[0], c1 = cc[1], c2 = cc[2], c3 = cc[3];
            double s0 = b0.x * c0.x, s1 = b0.y * c0.y;
            double s2 = b1.x * c1.x, s3 = b1.y * c1.y;
            double s4 = b2.x * c2.x, s5 = b2.y * c2.y;
            double s6 = b3.x * c3.x, s7 = b3.y * c3.y;
            double prev = (i0 > 0) ? b[i0 - 1] * c[i0 - 1] : 0.0;
            double r0 = s0 + prev; prev = s0;
            double r1 = s1 + prev; prev = s1;
            double r2 = s2 + prev; prev = s2;
            double r3 = s3 + prev; prev = s3;
            double r4 = s4 + prev; prev = s4;
            double r5 = s5 + prev; prev = s5;
            double r6 = s6 + prev; prev = s6;
            double r7 = s7 + prev;
            double2* aa = (double2*)(a + i0);
            aa[0] = make_double2(r0, r1);
            aa[1] = make_double2(r2, r3);
            aa[2] = make_double2(r4, r5);
            aa[3] = make_double2(r6, r7);
        } else {
            double prev = (i0 > 0) ? b[i0 - 1] * c[i0 - 1] : 0.0;
            for (long long i = i0; i < n; ++i) {
                double s = b[i] * c[i];
                a[i] = s + prev;
                prev = s;
            }
        }
    }
}

void tsvc_2_s252_launch(double* a, const double* b, const double* c, int64_t n) {
    if (n <= 0) return;
    const int T = 256;
    long long vthreads = (n + 7) / 8;
    long long blocks = (vthreads + T - 1) / T;
    const long long MAX_BLOCKS = 1u << 20;
    if (blocks > MAX_BLOCKS) blocks = MAX_BLOCKS;
    s252_v8<<<(unsigned)blocks, T, 0, 0>>>(a, b, c, (long long)n);
}
