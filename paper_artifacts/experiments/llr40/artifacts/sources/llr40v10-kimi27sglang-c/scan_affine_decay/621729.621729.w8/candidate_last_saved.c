#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>

/* Non-temporal scalar store.  Used for the first write pass over y so that
 * an array which is only written does not trigger write-allocate traffic. */
static inline void nt_store(double *restrict p, double v) {
    __asm__ volatile ("movntsd %0, %1" :: "x"(v), "m"(*p) : "memory");
}

void scan_affine_decay_fp64(const double *restrict c, const double *restrict x,
                            double *restrict y, const int64_t LEN_1D) {
    if (LEN_1D <= 1) return;

    /* For small inputs the recurrence is latency-bound and the serial loop
     * avoids parallel overhead. */
    if (LEN_1D < 8192) {
        for (int64_t i = 1; i < LEN_1D; ++i) {
            y[i] = c[i] * y[i - 1] + x[i];
        }
        return;
    }

    const int64_t BLK = 2048;
    const int64_t n = LEN_1D - 1;
    const int64_t nblocks = (n + BLK - 1) / BLK;

    double *restrict trans = (double *)malloc(2 * nblocks * sizeof(double));
    if (trans == NULL) return;
    double *restrict A = trans;
    double *restrict B = trans + nblocks;

    /* Pass 1: per block compute the affine transform (A,B) that maps the
     * incoming value y[start-1] to y[end].  At the same time write the local
     * recurrence with a zero seed (the offset B_i) into y using non-temporal
     * stores so that the first write pass does not cause write-allocate
     * traffic. */
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < nblocks; ++b) {
        int64_t i0 = 1 + b * BLK;
        int64_t ie = i0 + BLK - 1;
        if (ie >= LEN_1D) ie = LEN_1D - 1;

        double a = 1.0;
        double bb = 0.0;
        for (int64_t i = i0; i <= ie; ++i) {
            double ci = c[i];
            a *= ci;
            bb = bb * ci + x[i];
            nt_store(&y[i], bb);
        }
        A[b] = a;
        B[b] = bb;
    }
    _mm_sfence();

    /* Pass 2: serial prefix scan over the block transforms to obtain the
     * true incoming value for every block.  nblocks is only tens of thousands,
     * so this sequential step is tiny. */
    double *restrict prev = (double *)malloc(nblocks * sizeof(double));
    if (prev == NULL) {
        free(trans);
        return;
    }
    const double y0 = y[0];
    prev[0] = y0;
    for (int64_t b = 1; b < nblocks; ++b) {
        prev[b] = A[b - 1] * prev[b - 1] + B[b - 1];
    }

    /* Pass 3: re-apply the coefficient products to add the correct incoming
     * contribution to every element.  x is not read here. */
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < nblocks; ++b) {
        int64_t i0 = 1 + b * BLK;
        int64_t ie = i0 + BLK - 1;
        if (ie >= LEN_1D) ie = LEN_1D - 1;

        double a = 1.0;
        const double p = prev[b];
        for (int64_t i = i0; i <= ie; ++i) {
            a *= c[i];
            y[i] += a * p;
        }
    }

    free(prev);
    free(trans);
}
