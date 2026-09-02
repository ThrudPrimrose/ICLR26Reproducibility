#include <stdint.h>
#include <omp.h>

static inline int64_t i64min(int64_t a,int64_t b){return a<b?a:b;}

void wf_triangular_fp64(double *a, const int64_t n, uint8_t *workspace, const int64_t workspace_bytes)
{
    (void)workspace;(void)workspace_bytes;
    int64_t nt = omp_get_max_threads();
    if (nt < 1) nt = 1;
    // aim for ~4*nt blocks on the wide anti-diagonal -> good load balance
    int64_t T = n / (4*nt);
    if (T < 128) T = 128;
    if (T > 512) T = 512;
    if (T > n) T = n;
    const int64_t nb = (n + T - 1) / T;
    #pragma omp parallel
    for (int64_t d = 0; d <= 2*(nb-1); d++) {
        int64_t bi_lo = (d > nb-1) ? (d - (nb-1)) : 0;
        int64_t bi_hi = (d < nb-1) ? d : (nb-1);
        if (bi_lo > bi_hi) continue;
        #pragma omp for schedule(static)
        for (int64_t bi = bi_lo; bi <= bi_hi; bi++) {
            const int64_t bj = d - bi;
            const int64_t r0 = bi*T;
            const int64_t r1 = i64min((bi+1)*T, n);
            const int64_t c0 = bj*T;
            const int64_t c1 = i64min((bj+1)*T, n);
            for (int64_t i = (r0>1?r0:1); i < r1; i++) {
                int64_t js = c0 > i ? c0 : i;
                if (js >= c1) continue;
                double *row = a + i*n;
                double *prev = a + (i-1)*n;
                for (int64_t j = js; j < c1; j++)
                    row[j] = row[j] + prev[j] + row[j-1];
            }
        }
    }
}
