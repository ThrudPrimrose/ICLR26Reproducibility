/*
 * Optimized implementation of the TSVC_2 s319 microkernel for an OpenMP target offload arm.
 *
 * The reference implementation (hand port) is in /shared/tasks/tsvc_2_s319/tsvc_2_s319_reference.c.
 * This version adds explicit OpenMP target offloading and a parallel reduction to exploit the GPU.
 * The algorithm is mathematically equivalent to the reference:
 *   sum = Σ_i (c[i] + d[i] + c[i] + e[i])
 *   a[i] = c[i] + d[i]
 *   b[i] = c[i] + e[i]
 *   b[0] = sum
 * None of the input arrays alias each other (restrict) and the loop has no cross‑iteration dependencies.
 *
 * Offloading strategy:
 *   * Use a `target data` region to map the input and output arrays plus a scalar `sum` to/from the device.
 *   * Within the data region, launch a `teams distribute parallel for` loop with a `reduction(+ : sum)`.
 *   * The reduction is performed on the device, and the resulting sum is mapped back to the host.
 *   * After the offload, we store the sum into b[0] on the host to match the reference semantics.
 */

#include <stdint.h>

/*
 * Signature follows the C‑ABI expected by the benchmark harness.
 *   a   – output array, a[i] = c[i] + d[i]
 *   b   – output array, b[i] = c[i] + e[i]; b[0] overwritten with the sum
 *   c,d,e – input arrays
 *   LEN_1D – number of elements in each array
 */
void tsvc_2_s319_fp64(double *restrict a,
                      double *restrict b,
                      const double *restrict c,
                      const double *restrict d,
                      const double *restrict e,
                      const int64_t LEN_1D)
{
    double sum = 0.0;

    /*
     * Offload the work to the device. The `target data` clause maps the entire arrays and the
     * scalar `sum`. Inside the region a `target teams distribute parallel for` loop performs the
     * computation and reduces `sum` across all threads on the device.
     */
    #pragma omp target data map(to: c[0:LEN_1D], d[0:LEN_1D], e[0:LEN_1D]) \
                            map(tofrom: a[0:LEN_1D], b[0:LEN_1D], sum)
    {
        #pragma omp target teams distribute parallel for reduction(+:sum)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            double ai = c[i] + d[i];
            a[i] = ai;
            double bi = c[i] + e[i];
            b[i] = bi;
            sum += ai + bi;
        }
    }

    /*
     * The reference writes the final sum into b[0] after the loop. The device already wrote a value
     * to b[0] during the iteration, but we must replace it with the accumulated sum, so we do it on
     * the host after the offload completes.
     */
    b[0] = sum;
}

