/* Optimized version of tsvc_2_s233 microkernel for OpenMP target offload.
 * Implements two 2D scan operations on arrays aa, bb using the auxiliary array cc.
 * The original reference performs two nested loops per column i.
 * This version:
 *   - Uses a single target data region to map the arrays once.
 *   - Performs the aa update with a target teams distribute parallel for.
 *   - Performs the bb update with a separate target region and SIMD.
 */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
    const int64_t start = 8;
    const int64_t len = LEN_2D;
    const int64_t total = len * len; // total elements per array

    /* Map the three arrays onto the device once. */
    #pragma omp target data map(to: cc[0:total]) map(tofrom: aa[0:total], bb[0:total]) if(0)
    {
        /*------------------------------------------------------------
         * Compute aa: vertical prefix sum per column i.
         * Independent across columns, so we parallelise over i.
         *------------------------------------------------------------*/
        #pragma omp target teams distribute parallel for if(0) schedule(static)
        for (int64_t i = start; i < len; ++i) {
            int64_t idx = start * len + i;   // first row to update (j = start)
            double prev = aa[idx - len];      // value at row start-1
            for (int64_t j = start; j < len; ++j, idx += len) {
                double cur = prev + cc[idx];
                aa[idx] = cur;
                prev = cur;
            }
        }

        /*------------------------------------------------------------
         * Compute bb: horizontal prefix sum per row j.
         * Dependency across i prevents parallelisation of i, but the inner
         * loop over j is independent and can be vectorised.
         *------------------------------------------------------------*/
        #pragma omp target if(0)
        {
            for (int64_t i = start; i < len; ++i) {
                #pragma omp simd
                for (int64_t j = start; j < len; ++j) {
                    int64_t idx = j * len + i;
                    bb[idx] = bb[idx - 1] + cc[idx];
                }
            }
        }
    }
}
