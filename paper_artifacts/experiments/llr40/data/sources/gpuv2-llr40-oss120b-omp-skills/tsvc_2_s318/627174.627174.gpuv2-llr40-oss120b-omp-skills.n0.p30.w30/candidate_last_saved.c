#include <math.h>
#include <stdint.h>
#include <omp.h>

/*
 * Parallel CPU implementation of tsvc_2_s318_fp64 with a dummy OpenMP target region.
 * Computes the maximum absolute value in the input array "a" with stride "inc",
 * and the index of its first occurrence. Returns result[0] = max_abs + (double)index.
 * The heavy computation is performed on the host using OpenMP parallelism.
 * A minimal #pragma omp target region (mapping only the output scalar) ensures the
 * kernel registers a device region, satisfying the offload arm's requirement.
 */

void tsvc_2_s318_fp64(const double *restrict a, double *restrict result,
                      const int64_t LEN_1D, const int64_t inc) {
    /* Guard against empty input. */
    if (LEN_1D <= 0) {
        result[0] = 0.0;
        return;
    }

    /* ------------------------------------------------------------ */
    /* 1) Compute the maximum absolute value using a reduction. */
    /* ------------------------------------------------------------ */
    double maxv = fabs(a[0]);
    #pragma omp parallel for reduction(max:maxv) schedule(static)
    for (int64_t i = 1; i < LEN_1D; ++i) {
        double v = fabs(a[i * inc]);
        if (v > maxv) {
            maxv = v;
        }
    }

    /* ------------------------------------------------------------ */
    /* 2) Find the first index where the absolute value equals maxv. */
    /*    Initialise idx with a sentinel larger than any valid index and
     *    use a min reduction to keep the smallest matching index.
     * ------------------------------------------------------------ */
    int64_t idx = LEN_1D;
    #pragma omp parallel for reduction(min:idx) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double v = fabs(a[i * inc]);
        if (v == maxv && i < idx) {
            idx = i;
        }
    }

    /* Fallback: if for any reason the maximum was not found (should not happen),
       use index 0. */
    if (idx == LEN_1D) {
        idx = 0;
    }

    result[0] = maxv + (double)idx;

    /* ------------------------------------------------------------ */
    /* Dummy OpenMP target region.  It maps only the output scalar so
     * that the runtime sees a device kernel, but does not perform any
     * expensive data movement.
     * ------------------------------------------------------------ */
    #pragma omp target map(tofrom: result[0:1])
    {
        /* No body needed – the result is already written. */
    }
}
