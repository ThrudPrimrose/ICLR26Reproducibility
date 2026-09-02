#include <stdint.h>
#include <omp.h>

/*
 * ext_break_find_first kernel (TSVC s481)
 *
 * For each index i from 0 to LEN_1D-1, if d[i] < 0 the loop terminates before
 * executing the body.  Otherwise the body updates a[i] = a[i] + b[i] * c[i].
 * Only the first negative element in d causes an early exit; all prior
 * elements are processed.  The reference implementation in Python performs a
 * sequential scan, but we parallelise the search for the break index and then
 * apply the computation up to that point.
 */

void ext_break_find_first_fp64(double *restrict a,
                               const double *restrict b,
                               const double *restrict c,
                               const double *restrict d,
                               int64_t LEN_1D,
                               uint8_t *restrict workspace,
                               int64_t workspace_bytes)
{
    // The workspace is unused for this kernel.
    (void)workspace;
    (void)workspace_bytes;

    // Sentinel index indicating "no break" – one past the end.
    int64_t break_idx = LEN_1D;

    // Parallel phase: each thread scans its chunk for the first negative d.
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        // Compute a (approximately) equal chunk size.
        int64_t chunk = (LEN_1D + nthreads - 1) / nthreads; // ceil division
        int64_t i_start = tid * chunk;
        int64_t i_end = i_start + chunk;
        if (i_end > LEN_1D) i_end = LEN_1D;

        int64_t local_break = LEN_1D; // sentinel for this thread
        for (int64_t i = i_start; i < i_end; ++i) {
            if (d[i] < 0.0) {
                local_break = i;
                break; // first negative in this chunk
            }
        }
        // Reduce to the global minimum break index.
        #pragma omp critical
        {
            if (local_break < break_idx) {
                break_idx = local_break;
            }
        }
    }

    // Now apply the computation up to (but not including) break_idx.
    // If break_idx == 0 nothing is done; if break_idx == LEN_1D we process the
    // whole array.
    #pragma omp parallel for schedule(static) // vectorise inside
    for (int64_t i = 0; i < break_idx; ++i) {
        a[i] = a[i] + b[i] * c[i];
    }
}
