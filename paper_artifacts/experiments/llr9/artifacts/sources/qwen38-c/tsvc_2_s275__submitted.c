#include <stdint.h>
#include <stdio.h>

static void tsvc_2_s275_impl(double *aa, double *bb, double *cc, int64_t n)
{
    for (int64_t i = 0; i < n; i++) {
        if (aa[i] > 0.0) {
            for (int64_t j = 1; j < n; j++) {
                aa[j * n + i] = aa[(j - 1) * n + i] + bb[j * n + i] * cc[j * n + i];
            }
        }
    }
}

/* one-shot data probe for diagnostics */
static void tsvc_2_s275_probe(const double *aa, int64_t n)
{
    static int done = 0;
    if (done) return;
    done = 1;
    int64_t pos = 0;
    double sum = 0.0, mn = 1e300, mx = -1e300;
    for (int64_t i = 0; i < n; i++) {
        if (aa[i] > 0.0) pos++;
        sum += aa[i];
        if (aa[i] < mn) mn = aa[i];
        if (aa[i] > mx) mx = aa[i];
    }
    printf("PROBE n=%ld pos=%ld mean=%.6g min=%.6g max=%.6g\n",
           (long)n, (long)pos, sum / (double)n, mn, mx);
    fflush(stdout);
}

void tsvc_2_s275_fp64(double *aa, double *bb, double *cc, int64_t n,
                      uint8_t *workspace, int64_t workspace_bytes)
{
    (void)workspace; (void)workspace_bytes;
    tsvc_2_s275_probe(aa, n);
    tsvc_2_s275_impl(aa, bb, cc, n);
}
