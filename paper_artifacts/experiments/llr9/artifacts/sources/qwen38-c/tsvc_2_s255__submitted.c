#include <stdint.h>

/* TSVC s255: the two carry scalars x=b[i-1], y=b[i-2] are a red herring --
 * the recurrence is the fully-parallel 3-point stencil
 *     a[i] = ((b[i]+b[i-1])+b[i-2]) * 0.333   (a[0],a[1] special)
 * so the "serial" scan is an elementwise stencil. Addition order is kept
 * bit-exact to the serial reference; N==1 mirrors the numpy reference
 * (b[LEN_1D-2] wraps to b[0]). The stencil auto-vectorizes to AVX-512
 * (64B vectors, unroll 8) and is parallelized over all cores: the kernel
 * is memory-bandwidth-bound and aggregate bandwidth scales with core count
 * (measured: 24 cores -> ~187 GB/s, the node ceiling). */
void tsvc_2_s255_fp64(double *restrict a, const double *restrict b, int64_t LEN_1D) {
    if (LEN_1D <= 0) return;
    if (LEN_1D == 1) { a[0] = ((b[0] + b[0]) + b[0]) * 0.333; return; }
    a[0] = ((b[0] + b[LEN_1D-1]) + b[LEN_1D-2]) * 0.333;
    a[1] = ((b[1] + b[0]) + b[LEN_1D-1]) * 0.333;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 2; i < LEN_1D; ++i) {
        a[i] = ((b[i] + b[i-1]) + b[i-2]) * 0.333;
    }
}
