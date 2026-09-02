/*
 * TSVC 2 kernel "s255"
 *
 * Reference implementation (numpy) computes for each i:
 *   a[i] = (b[i] + x + y) * 0.333
 * where x and y are the two preceding elements of b, with wrap‑around
 * (i‑1 and i‑2 modulo LEN_1D). The original loop updates x and y
 * sequentially, creating a dependence chain. By expanding the recurrence
 * we can rewrite the computation as an independent three‑point stencil:
 *
 *   a[i] = (b[i] + b[(i-1+LEN)%LEN] + b[(i-2+LEN)%LEN]) * 0.333
 *
 * This formulation is fully parallel and vectorisable. The implementation
 * below handles the two boundary elements explicitly and then processes the
 * remaining range with an OpenMP parallel for. The loop body is simple and
 * the compiler is able to auto‑vectorise; the "simd" clause is added to make
 * the intent explicit.
 */

#include <stdint.h>
#include <omp.h>

/*
 * Signature required by the benchmark harness.
 *   a       – output array, length LEN_1D
 *   b       – input array, length LEN_1D (read‑only)
 *   LEN_1D  – number of elements (guaranteed >= 2)
 */
void tsvc_2_s255_fp64(double *restrict a, const double *restrict b, int64_t LEN_1D, uint8_t *restrict workspace, int64_t workspace_bytes)
{
    const double factor = 0.333; // matches the reference constant

    if (LEN_1D <= 0)
        return;

    /* Handle the first two elements which need wrap‑around indices. */
    if (LEN_1D >= 1) {
        a[0] = (b[0] + b[LEN_1D - 1] + b[LEN_1D - 2]) * factor;
    }
    if (LEN_1D >= 2) {
        a[1] = (b[1] + b[0] + b[LEN_1D - 1]) * factor;
    }

    /* The rest of the elements have a straight‑forward stencil with unit stride. */
    if (LEN_1D > 2) {
        /* Parallelise over the outer loop; each iteration is independent. */
        #pragma omp parallel for simd schedule(static)
        for (int64_t i = 2; i < LEN_1D; ++i) {
            a[i] = (b[i] + b[i - 1] + b[i - 2]) * factor;
        }
    }
}

