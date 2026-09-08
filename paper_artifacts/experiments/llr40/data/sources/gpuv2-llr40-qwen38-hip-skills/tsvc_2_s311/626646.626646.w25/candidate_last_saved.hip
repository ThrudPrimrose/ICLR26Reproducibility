/* TSVC tsvc_2 s311: sum a[0..LEN_1D-1] into sum_out[0] (fp64).
 *
 * Two small kernels, deliberately written in the most compiler-boring form:
 *  k1: grid-stride pass with 8 fp64 accumulators per thread, plain
 *      shared-memory tree per block -> one partial per block.
 *  k2: single block sums the partials -> sum_out[0].
 * No shuffles, no 64-bit integer division in the inner path, no atomics,
 * fully deterministic (fixed order) output.
 */
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define T2S311_BLOCK 256
#define T2S311_UNROLL 8
#define T2S311_CAP 4096

#define HIPCHK(x)                                                                  \
    do {                                                                           \
        hipError_t e_ = (x);                                                       \
        if (e_ != hipSuccess) {                                                    \
            std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), \
                         __FILE__, __LINE__);                                      \
            std::abort();                                                          \
        }                                                                          \
    } while (0)

__global__ void __launch_bounds__(T2S311_BLOCK)
    t2s311_partials(const double * __restrict__ a, double * __restrict__ parts,
                    const int64_t n) {
    const int64_t gtid = (int64_t)blockIdx.x * T2S311_BLOCK + threadIdx.x;
    const int64_t chunk = (int64_t)gridDim.x * T2S311_BLOCK * T2S311_UNROLL;

    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    double s4 = 0.0, s5 = 0.0, s6 = 0.0, s7 = 0.0;

    int64_t off = gtid * T2S311_UNROLL;
    const int64_t full_end = n - (T2S311_UNROLL - 1);
    while (off < full_end) {
        s0 += a[off + 0];
        s1 += a[off + 1];
        s2 += a[off + 2];
        s3 += a[off + 3];
        s4 += a[off + 4];
        s5 += a[off + 5];
        s6 += a[off + 6];
        s7 += a[off + 7];
        off += chunk;
    }
    if (off < n) {
        for (int i = 0; off + i < n; ++i) {
            double v = a[off + i];
            switch (i) {
                case 0: s0 += v; break;
                case 1: s1 += v; break;
                case 2: s2 += v; break;
                case 3: s3 += v; break;
                case 4: s4 += v; break;
                case 5: s5 += v; break;
                case 6: s6 += v; break;
                default: s7 += v; break;
            }
        }
    }

    double t = (s0 + s4) + (s1 + s5) + ((s2 + s6) + (s3 + s7));

    __shared__ double ld[T2S311_BLOCK];
    ld[threadIdx.x] = t;
    __syncthreads();
#pragma unroll
    for (int stride = T2S311_BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < (unsigned)stride)
            ld[threadIdx.x] += ld[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) parts[blockIdx.x] = ld[0];
}

__global__ void __launch_bounds__(T2S311_BLOCK)
    t2s311_final(const double * __restrict__ parts, double * __restrict__ out,
                 const int nb) {
    double t = 0.0;
    for (int i = threadIdx.x; i < nb; i += T2S311_BLOCK) t += parts[i];

    __shared__ double ld[T2S311_BLOCK];
    ld[threadIdx.x] = t;
    __syncthreads();
#pragma unroll
    for (int stride = T2S311_BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < (unsigned)stride)
            ld[threadIdx.x] += ld[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = ld[0];
}

static double *g_parts = nullptr; /* [0..T2S311_CAP) block partials */

static void ensure_scratch() {
    if (g_parts) return;
    HIPCHK(hipMalloc(&g_parts, (size_t)(T2S311_CAP + 1) * sizeof(double)));
}

__attribute__((constructor)) static void t2s311_prealloc() {
    /* Best effort; ensure_scratch() retries lazily on failure. */
    ensure_scratch();
}

extern "C" void tsvc_2_s311_fp64(const double *__restrict__ a,
                                 double *__restrict__ sum_out, const int64_t LEN_1D) {
    ensure_scratch();
    const int64_t n = LEN_1D;
    int64_t grid = (n + (int64_t)T2S311_BLOCK * T2S311_UNROLL - 1) /
                   ((int64_t)T2S311_BLOCK * T2S311_UNROLL);
    if (grid < 1) grid = 1;
    if (grid > T2S311_CAP) grid = T2S311_CAP;

    t2s311_partials<<<(int)grid, T2S311_BLOCK>>>(a, g_parts, n);
    HIPCHK(hipGetLastError());
    t2s311_final<<<1, T2S311_BLOCK>>>(g_parts, sum_out, (int)grid);
    HIPCHK(hipGetLastError());
    HIPCHK(hipDeviceSynchronize());
}
