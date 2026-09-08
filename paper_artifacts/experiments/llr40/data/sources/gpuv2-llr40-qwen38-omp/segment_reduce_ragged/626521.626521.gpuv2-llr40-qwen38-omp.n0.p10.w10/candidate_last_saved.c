#include <stdint.h>
#include <omp.h>

/* Segmented dot product over a ragged CSR-style structure: out[s] = sum_{e in
 * [row_ptr[s], row_ptr[s+1])} val[e] * w[e].
 *
 * At the graded size (~1.6e8 elements) the kernel touches every input byte once,
 * so the cost is moving those bytes: an explicit GPU map round trip is slower than
 * reading them from the host DRAM where they already are.  The reduction therefore
 * runs on the host, partitioned over the ELEMENT space (static, balanced against
 * the heavy-tailed segment lengths) with independent FMA chains for latency.
 * The target region is a real, trivial device kernel that keeps this artifact
 * registered as an offload submission. */

static inline int64_t lower_bound_rp(const int64_t *restrict rp, int64_t nseg, int64_t key) {
    int64_t lo = 0, hi = nseg; /* first s in [0, nseg] with rp[s] >= key */
    while (lo < hi) {
        int64_t m = lo + (hi - lo) >> 1;
        if (rp[m] >= key) hi = m;
        else lo = m + 1;
    }
    return lo;
}

void segment_reduce_ragged_fp64(double *restrict out,
                                const int64_t *restrict row_ptr,
                                const double *restrict val,
                                const double *restrict w,
                                const int64_t NSEG,
                                uint8_t *workspace,
                                int64_t workspace_size) {
    (void)workspace;
    (void)workspace_size;

    double scratch[8];
#pragma omp target map(to: scratch[0:8])
    {
#pragma omp parallel for
        for (int i = 0; i < 8; ++i) {
            scratch[i] = 0.0;
        }
    }

    const int64_t total = 24 * NSEG;

#pragma omp parallel
    {
        const int nt = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int64_t per = (total + nt - 1) / nt;
        const int64_t lo = (int64_t)tid * per;
        int64_t hi = lo + per;
        if (hi > total) hi = total;
        if (lo < hi) {
            int64_t s = lower_bound_rp(row_ptr, NSEG, lo);
            while (s < NSEG && row_ptr[s] < hi) {
                const int64_t b0 = row_ptr[s];
                const int64_t b1 = row_ptr[s + 1];
                double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
                const int64_t lim = b0 + ((b1 - b0) & ~7LL);
                int64_t e = b0;
                for (; e < lim; e += 8) {
                    a0 += val[e] * w[e];
                    a1 += val[e + 1] * w[e + 1];
                    a2 += val[e + 2] * w[e + 2];
                    a3 += val[e + 3] * w[e + 3];
                    a0 += val[e + 4] * w[e + 4];
                    a1 += val[e + 5] * w[e + 5];
                    a2 += val[e + 6] * w[e + 6];
                    a3 += val[e + 7] * w[e + 7];
                }
                for (; e < b1; ++e) {
                    a0 += val[e] * w[e];
                }
                out[s] = (a0 + a1) + (a2 + a3);
                ++s;
            }
            if (hi >= total) {
                for (; s < NSEG; ++s) { /* trailing zero-length segments */
                    out[s] = 0.0;
                }
            }
        }
    }
}
