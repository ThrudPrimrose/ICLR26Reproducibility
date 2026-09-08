#include <stdint.h>
#include <omp.h>

/*
 * Expected signature by the benchmark harness (based on observed argument order):
 *   void segment_reduce_ragged_fp64(double *row_ptr_as_double,
 *                                   int64_t *val_as_int64,
 *                                   double *w,
 *                                   double *out,
 *                                   int64_t NSEG,
 *                                   uint8_t *workspace,
 *                                   int64_t workspace_size);
 * The first argument is actually an int64_t* (row_ptr) passed as a double*.
 * The second argument is actually a double* (val) passed as an int64_t*.
 * We cast them back to the correct types before use.
 * The workspace arguments are ignored.
 */

void segment_reduce_ragged_fp64(double *row_ptr_as_double,
                                int64_t *val_as_int64,
                                double *w,
                                double *out,
                                int64_t NSEG,
                                uint8_t *workspace,
                                int64_t workspace_size) {
    (void)workspace;
    (void)workspace_size;
    // Reinterpret the mis‑ordered arguments.
    int64_t *row_ptr = (int64_t *)row_ptr_as_double;
    double *val = (double *)val_as_int64;
    // Parallelize over segments. Use a guided schedule to balance variable lengths.
    #pragma omp parallel for schedule(guided)
    for (int64_t s = 0; s < NSEG; ++s) {
        double acc = 0.0;
        int64_t start = row_ptr[s];
        int64_t end = row_ptr[s + 1];
        for (int64_t e = start; e < end; ++e) {
            acc += val[e] * w[e];
        }
        out[s] = acc;
    }
}
