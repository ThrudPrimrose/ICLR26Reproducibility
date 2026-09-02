/* TSVC tsvc_2_5: out[i] = tmp[i]*tmp[i+1], tmp[j] = a[j-1]+a[j]+a[j+1]
   Fused to one streaming pass:
     out[i] = (a[i-1]+a[i]+a[i+1]) * (a[i]+a[i+1]+a[i+2]),  i in [1, LEN_1D-3]
   out[0], out[n-2], out[n-1] are never written (matches the reference).
   Large n: vectorized (AVX-512) OpenMP loop, dynamic schedule tuned on the
   target node (bandwidth-bound: read a once + write out once).
   Small n: serial masked-SIMD loop (avoids both the thread spawn cost and
   alignment-versioned scalar fallback). */
#include <stdint.h>
#include <omp.h>

void fuse_stencil_through_transient_fp64(double *a, double *out, int64_t LEN_1D,
                                         uint8_t *workspace, int64_t workspace_bytes)
{
    (void)workspace; (void)workspace_bytes;
    if (LEN_1D < 4) return;
    double * __restrict o = out;
    const double * __restrict aa = a;
    int64_t lo = 1, hi = LEN_1D - 2; /* i in [lo, hi) */

    if (LEN_1D >= (1LL << 17)) {
        #pragma omp parallel for simd schedule(dynamic, 524288)
        for (int64_t i = lo; i < hi; i++)
            o[i] = (aa[i-1] + aa[i] + aa[i+1]) * (aa[i] + aa[i+1] + aa[i+2]);
    } else {
        #pragma omp simd
        for (int64_t i = lo; i < hi; i++)
            o[i] = (aa[i-1] + aa[i] + aa[i+1]) * (aa[i] + aa[i+1] + aa[i+2]);
    }
}
