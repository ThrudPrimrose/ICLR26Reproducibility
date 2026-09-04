#include <stdint.h>
#include <stdio.h>
#include <omp.h>

static void srr_impl(const int64_t *restrict row_ptr, const double *restrict val,
                     const double *restrict w, double *restrict out, const int64_t NSEG)
{
    if (NSEG <= 0) { printf("DBG: NSEG<=0 (%lld)\n", (long long)NSEG); fflush(stdout); return; }
    printf("DBG: NSEG=%lld total=row_ptr[NSEG]=%lld NSEG*24=%lld\n",
           (long long)NSEG, (long long)row_ptr[NSEG], (long long)(NSEG*24));
    printf("DBG: rp[0..3]=%lld %lld %lld %lld rp[NSEG-2..NSEG]=%lld %lld %lld\n",
           (long long)row_ptr[0], (long long)row_ptr[1], (long long)row_ptr[2], (long long)row_ptr[3],
           (long long)row_ptr[NSEG-2], (long long)row_ptr[NSEG-1], (long long)row_ptr[NSEG]);
    long long minn = (long long)1e18, maxn = (long long)-1; long long nzero = 0;
    for (int64_t s = 0; s < NSEG; ++s) {
        int64_t n = row_ptr[s+1] - row_ptr[s];
        if (n < minn) minn = n;
        if (n > maxn) maxn = n;
        if (n == 0) nzero++;
    }
    printf("DBG: lens min=%lld max=%lld nzero=%lld val[0..2]=%f %f %f w[0..2]=%f %f %f\n",
           minn, maxn, nzero, val[0], val[1], val[2], w[0], w[1], w[2]);
    printf("DBG: out pre: [0]=%f [1]=%f [348512]=%f\n", out[0], out[1], NSEG > 348513 ? out[348512] : -12345.0);
    /* the v1 algorithm */
    const int64_t total = row_ptr[NSEG];
    int nt = (int)omp_get_max_threads();
    if (nt < 1) nt = 1;
    if ((int64_t)nt > NSEG) nt = (int)NSEG;
    if ((int64_t)nt > total) nt = (int)total;
    #pragma omp parallel num_threads(nt)
    {
        const int64_t tid = omp_get_thread_num();
        const int64_t lo = (total * tid) / nt;
        const int64_t hi = (total * (tid + 1)) / nt;
        int64_t lo2 = 0, hi2 = NSEG;
        while (lo2 < hi2) { int64_t mid = (lo2 + hi2) >> 1; if (row_ptr[mid+1] <= lo) lo2 = mid+1; else hi2 = mid; }
        int64_t s_lo = lo2;
        lo2 = s_lo; hi2 = NSEG;
        while (lo2 < hi2) { int64_t mid = (lo2 + hi2) >> 1; if (row_ptr[mid+1] <= hi) lo2 = mid+1; else hi2 = mid; }
        int64_t s_hi = lo2;
        if (tid == 0) { for (int64_t s = 0; s < s_lo; ++s) out[s] = 0.0; }
        for (int64_t s = s_lo; s < s_hi; ++s) {
            const int64_t rs = row_ptr[s];
            int64_t n = row_ptr[s+1] - rs;
            double acc = 0.0;
            if (n > 0) {
                const double *pv = val + rs;
                const double *pw = w + rs;
                for (int64_t e = 0; e < n; ++e) acc += pv[e] * pw[e];
            }
            out[s] = acc;
        }
    }
    long long nz = 0; for (int64_t s = 0; s < NSEG; ++s) if (out[s] != 0.0) nz++;
    printf("DBG: nt used=%d post: out[0]=%f out[100]=%f out[348512]=%f nonzero=%lld/%lld\n",
           nt, out[0], out[100], NSEG > 348513 ? out[348512] : -12345.0, nz, (long long)NSEG);
    fflush(stdout);
}

void segment_reduce_ragged_fp64(const int64_t *restrict row_ptr, const double *restrict val,
                                const double *restrict w, double *restrict out, const int64_t NSEG)
{ srr_impl(row_ptr, val, w, out, NSEG); }

void segment_reduce_ragged(const int64_t *restrict row_ptr, const double *restrict val,
                           const double *restrict w, double *restrict out, const int64_t NSEG)
{ srr_impl(row_ptr, val, w, out, NSEG); }
