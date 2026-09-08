#include <stdint.h>
#include <stdio.h>

void segment_reduce_ragged_fp64(double val[], int64_t row_ptr[], double w[], double out[], int64_t NSEG, uint8_t *workspace, int64_t workspace_size)
{
    static int printed = 0;
    int dummy = 0;
    #pragma omp target map(tofrom: dummy)
    dummy = 1;
    if (!printed) {
        printed = 1;
        int64_t total = row_ptr[NSEG];
        int64_t maxlen = 0;
        for (int64_t s = 0; s < NSEG; s++) {
            int64_t l = row_ptr[s+1] - row_ptr[s];
            if (l > maxlen) maxlen = l;
        }
        printf("PROBE NSEG=%lld total=%lld maxlen=%lld ws=%lld lens[0..3]=%lld,%lld,%lld,%lld v0=%g w0=%g out0_in=%g\n",
            (long long)NSEG, (long long)total, (long long)maxlen, (long long)workspace_size,
            (long long)(row_ptr[1]-row_ptr[0]), (long long)(row_ptr[2]-row_ptr[1]),
            (long long)(row_ptr[3]-row_ptr[2]), (long long)(row_ptr[4]-row_ptr[3]),
            val[0], w[0], out[0]);
        fflush(stdout);
    }
    for (int64_t s = 0; s < NSEG; s++) {
        double acc = 0.0;
        for (int64_t e = row_ptr[s]; e < row_ptr[s+1]; e++) acc += val[e] * w[e];
        out[s] = acc;
    }
}
