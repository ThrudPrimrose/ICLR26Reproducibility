/* hpcagent_bench stub headers -- keep as given */
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* TSVC s4112:  a[i] = a[i] + b[ip[i]] * 2.0
 * ABI: a,b = double[LEN]; ip = int32 index array; LEN_1D; workspace.
 * No loop-carried dependence (each i writes only a[i], reads a[i],b[ip[i]],ip[i])
 * => safe to thread i and to simd the lanes. */

void tsvc_2_s4112_fp64(double *restrict a,
                       const double *restrict b,
                       const int32_t *restrict ip,
                       const int64_t LEN_1D,
                       uint8_t *restrict workspace,
                       const int64_t workspace_size)
{
    (void)workspace;
    (void)workspace_size;
    const int64_t n = LEN_1D;
    if (n <= 0) return;

    const int nt = omp_get_max_threads();
    if (nt <= 1 || n < (1 << 16)) {
        for (int64_t i = 0; i < n; i++)
            a[i] += 2.0 * b[(int64_t)ip[i]];
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++)
        a[i] += 2.0 * b[(int64_t)ip[i]];
}
