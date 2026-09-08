// tsvc_2_s3110: find max of LEN_2D x LEN_2D double array (first occurrence on
// ties, row-major), write maxv + xindex + yindex to bb[0].
// HIP implementation: vectorized two-stage device reduction.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstddef>
#include <climits>

#define NT 256
#define MAXB 8192

struct __align__(16) Pair {
    double v;
    long long idx;
};

__device__ __forceinline__ Pair ident() {
    Pair p;
    p.v = NAN;
    p.idx = LLONG_MAX;
    return p;
}

// Combine (v,idx) with (ov,oidx). Total order: value (NaN lowest), then smaller
// linear index. Result is the unique element that a sequential scan with strict
// '>' would leave as champion, for any ordering of inputs.
__device__ __forceinline__ void merge_best(double &v, long long &idx, double ov, long long oidx) {
    if (ov > v) { v = ov; idx = oidx; return; }
    if (v > ov) return;
    // equal values or NaN involved: NaN loses to non-NaN; else smaller index wins
    bool mynan = (v != v), othnan = (ov != ov);
    bool better;
    if (mynan != othnan) better = mynan;
    else better = (oidx < idx);
    if (better) { v = ov; idx = oidx; }
}

__device__ __forceinline__ Pair warp_merge(Pair x) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        double ov = __shfl_down_sync(0xffffffffffffffffull, x.v, o);
        long long oi = __shfl_down_sync(0xffffffffffffffffull, x.idx, o);
        merge_best(x.v, x.idx, ov, oi);
    }
    return x;
}

__device__ __forceinline__ Pair block_reduce(Pair x) {
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int nwarp = blockDim.x >> 5;
    x = warp_merge(x);
    if (nwarp > 1) {
        __shared__ Pair sm[NT / 32];
        if (lane == 0) sm[wid] = x;
        __syncthreads();
        if (wid == 0) {
            Pair y = (lane < nwarp) ? sm[lane] : ident();
            y = warp_merge(y);
            if (lane == 0) x = y;
        }
    }
    return x;
}

__global__ void reduceK(const double4 *__restrict__ av, const double *__restrict__ a,
                        int64_t nvec, int64_t ntot, Pair *__restrict__ part) {
    double v = NAN;
    long long idx = LLONG_MAX;
    // tail (<4 elements) handled by block 0, thread 0
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int64_t k = nvec * 4; k < ntot; ++k) merge_best(v, idx, a[k], k);
    }
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < nvec; i += stride) {
        double4 d = av[i];
        int64_t b = i * 4;
        merge_best(v, idx, d.x, b);
        merge_best(v, idx, d.y, b + 1);
        merge_best(v, idx, d.z, b + 2);
        merge_best(v, idx, d.w, b + 3);
    }
    Pair x = block_reduce(Pair{v, idx});
    if (threadIdx.x == 0) part[blockIdx.x] = x;
}

__global__ void finalK(const Pair *__restrict__ part, int64_t nparts,
                       const double *__restrict__ a, int64_t N,
                       double *__restrict__ bb) {
    double v = NAN;
    long long idx = LLONG_MAX;
    for (int64_t i = threadIdx.x; i < nparts; i += blockDim.x) {
        merge_best(v, idx, part[i].v, part[i].idx);
    }
    Pair x = block_reduce(Pair{v, idx});
    if (threadIdx.x == 0) {
        double a0 = a[0];
        double chk;
        if (a0 != a0) {
            chk = a0; // reference stays at (NaN,0,0) when the first element is NaN
        } else {
            long long xi = idx / N;
            long long yi = idx - xi * N;
            chk = v + (double)xi;
            chk = chk + (double)yi;
        }
        bb[0] = chk;
    }
}

static Pair *d_part = nullptr;

extern "C" void tsvc_2_s3110_fp64(const double *__restrict__ aa, double *__restrict__ bb,
                                  const int64_t LEN_2D) {
    if (!d_part) {
        hipError_t e = hipMalloc((void **)&d_part, (size_t)MAXB * sizeof(Pair));
        if (e != hipSuccess) return;
    }
    int64_t ntot = LEN_2D * LEN_2D;
    int64_t nvec = ntot >> 2;
    int64_t blocks = (nvec + NT - 1) / NT;
    if (blocks > MAXB) blocks = MAXB;
    if (blocks < 1) blocks = 1;
    reduceK<<<(unsigned)blocks, NT>>>((const double4 *)aa, aa, nvec, ntot, d_part);
    finalK<<<1, NT>>>(d_part, blocks, aa, LEN_2D, bb);
}
