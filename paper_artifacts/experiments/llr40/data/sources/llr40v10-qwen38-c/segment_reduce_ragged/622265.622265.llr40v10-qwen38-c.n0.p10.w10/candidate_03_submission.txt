#include <stdint.h>
#include <omp.h>

static inline double dot_seg(const double *restrict pv, const double *restrict pw, int64_t n)
{
    double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
    double a4 = 0.0, a5 = 0.0, a6 = 0.0, a7 = 0.0;
    int64_t e = 0;
    for (; e + 8 <= n; e += 8) {
        a0 += pv[e+0] * pw[e+0];
        a1 += pv[e+1] * pw[e+1];
        a2 += pv[e+2] * pw[e+2];
        a3 += pv[e+3] * pw[e+3];
        a4 += pv[e+4] * pw[e+4];
        a5 += pv[e+5] * pw[e+5];
        a6 += pv[e+6] * pw[e+6];
        a7 += pv[e+7] * pw[e+7];
    }
    for (; e < n; ++e) a0 += pv[e] * pw[e];
    return ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
}

static inline void seg_range(double *restrict out, const int64_t *restrict row_ptr,
                             const double *restrict val, const double *restrict w,
                             int64_t s_lo, int64_t s_hi)
{
    int64_t start = row_ptr[s_lo];
    for (int64_t s = s_lo; s < s_hi; ++s) {
        int64_t end = row_ptr[s+1];
        out[s] = dot_seg(val + start, w + start, end - start);
        start = end;
    }
}

void segment_reduce_ragged_fp64(double *restrict out, const int64_t *restrict row_ptr,
                                const double *restrict val, const double *restrict w,
                                const int64_t NSEG)
{
    if (NSEG <= 0) return;
    const int64_t total = row_ptr[NSEG];
    int nt = (int)omp_get_max_threads();
    if (nt < 1) nt = 1;
    if ((int64_t)nt > NSEG) nt = (int)NSEG;
    if ((int64_t)nt > total) nt = (int)total;
    if (nt <= 1) {
        seg_range(out, row_ptr, val, w, 0, NSEG);
        return;
    }
    #pragma omp parallel num_threads(nt)
    {
        const int64_t tid = omp_get_thread_num();
        const int nt2 = omp_get_num_threads();
        const int64_t lo = (total * tid) / nt2;
        const int64_t hi = (total * (tid + 1)) / nt2;
        int64_t a = 0, b = NSEG;
        while (a < b) { int64_t m = (a + b) >> 1; if (row_ptr[m+1] <= lo) a = m+1; else b = m; }
        int64_t s_lo = a;
        a = s_lo; b = NSEG;
        while (a < b) { int64_t m = (a + b) >> 1; if (row_ptr[m+1] <= hi) a = m+1; else b = m; }
        int64_t s_hi = a;
        seg_range(out, row_ptr, val, w, s_lo, s_hi);
    }
}
