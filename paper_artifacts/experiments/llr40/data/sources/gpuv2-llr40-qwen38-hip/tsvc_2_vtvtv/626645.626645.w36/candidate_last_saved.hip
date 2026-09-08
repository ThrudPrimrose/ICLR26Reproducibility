// TSVC tsvc_2/vtvtv: a[i] = a[i] * b[i] * c[i]   (fp64, memory-bandwidth bound)
#include <hip/hip_runtime.h>
#include <cstdint>

__global__ void vtvtv_kernel(double* __restrict__ a,
                             const double* __restrict__ b,
                             const double* __restrict__ c,
                             const int64_t n) {
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    // alignment: make the vector body start on a 32B boundary for all 3 arrays
    uintptr_t m = (((uintptr_t)a ^ (uintptr_t)b) ^ (uintptr_t)c) & 31u;
    int64_t head = m ? ((int64_t)(32 - m) >> 3) : 0;
    if (head > n) head = n;

    if (head && blockIdx.x == 0) {
        for (int64_t i = threadIdx.x; i < head; i += blockDim.x)
            a[i] = a[i] * b[i] * c[i];
    }

    const int64_t nv = (n - head) >> 2;   // number of double4 elements
    const int64_t tail = (n - head) - (nv << 2);

    double4* a4 = (double4*)(a + head);
    const double4* b4 = (double4*)(b + head);
    const double4* c4 = (double4*)(c + head);

    for (int64_t i = tid; i < nv; i += stride) {
        double4 av = a4[i];
        double4 bv = b4[i];
        double4 cv = c4[i];
        double4 r;
        r.x = av.x * bv.x * cv.x;
        r.y = av.y * bv.y * cv.y;
        r.z = av.z * bv.z * cv.z;
        r.w = av.w * bv.w * cv.w;
        a4[i] = r;
    }

    if (tail && blockIdx.x == 0) {
        const int64_t s = head + (nv << 2);
        for (int64_t i = threadIdx.x; i < tail; i += blockDim.x)
            a[s + i] = a[s + i] * b[s + i] * c[s + i];
    }
}

extern "C" void tsvc_2_vtvtv_fp64(double* a, const double* b, const double* c, const int64_t n) {
    if (n <= 0) return;
    const int block = 256;
    int64_t want = (n / 4 + block - 1) / block;
    const int64_t cap = 228 * 8;   // 1824 blocks
    unsigned grid = (unsigned)(want < cap ? want : cap);
    vtvtv_kernel<<<grid, block>>>(a, b, c, n);
}
