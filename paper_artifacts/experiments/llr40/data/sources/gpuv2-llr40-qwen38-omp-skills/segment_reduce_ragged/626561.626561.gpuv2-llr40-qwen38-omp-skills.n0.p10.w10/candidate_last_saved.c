#include <stdint.h>
#include <omp.h>

void segment_reduce_ragged_fp64(double *out, const int64_t *row_ptr,
                                const double *val, const double *w,
                                int64_t NSEG, uint8_t *workspace,
                                int64_t workspace_size)
{
    (void)workspace;
    (void)workspace_size;
    if (NSEG <= 0)
        return;
    const int64_t N = row_ptr[NSEG];
    if (N <= 0)
        return;

    /* Small problems: the GPU round-trip (launch + HtoD/DtoH copies) costs far
     * more than a serial host sweep.  Stay on the host below this size. */
    if (N < 10000000) {
        for (int64_t s = 0; s < NSEG; s++) {
            double acc = 0.0;
            const int64_t e0 = row_ptr[s];
            const int64_t e1 = row_ptr[s + 1];
            for (int64_t e = e0; e < e1; e++)
                acc += val[e] * w[e];
            out[s] = acc;
        }
        return;
    }

    /* Large problems: offload.  The APU copy is charged to us, but the GPU
     * clears 16*N bytes far faster than one CPU core can. */
#pragma omp target data map(to: row_ptr[0:NSEG + 1]) map(to: val[0:N]) \
    map(to: w[0:N]) map(from: out[0:NSEG])
    {
#pragma omp target teams distribute parallel for
        for (int64_t s = 0; s < NSEG; s++) {
            double acc = 0.0;
            const int64_t e0 = row_ptr[s];
            const int64_t e1 = row_ptr[s + 1];
            for (int64_t e = e0; e < e1; e++)
                acc += val[e] * w[e];
            out[s] = acc;
        }
    }
}
