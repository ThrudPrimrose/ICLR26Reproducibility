#include <stdint.h>

/*
 * Optimized version of ext_break_capture kernel.
 * Finds the first element in array `a` that is greater than constant K=1.0,
 * writes its index to out_index[0] and its value to out_value[0].
 * If no such element exists, writes -1 and -1.0.
 *
 * Parallelized with OpenMP reduction to compute the minimal index satisfying the condition.
 * For small arrays, sequential fallback avoids parallel overhead.
 */

void ext_break_capture_fp64(const double *restrict a,
                            int64_t *restrict out_index,
                            double *restrict out_value,
                            const int64_t LEN_1D) {
    const double k = 1.0;
    // Small‑size fallback: avoid parallel overhead for tiny inputs.
    if (LEN_1D < 1024) {
        // Sequential scan identical to reference.
        out_index[0] = -1;
        out_value[0] = -1.0;
        for (int64_t i = 0; i < LEN_1D; ++i) {
            if (a[i] > k) {
                out_index[0] = i;
                out_value[0] = a[i];
                break;
            }
        }
        return;
    }

    // Parallel reduction to find minimal index where a[i] > k.
    // OpenMP reduction(min:) initializes each thread's copy with the
    // maximum representable value; we later test against LEN_1D.
    int64_t min_idx = LEN_1D; // sentinel larger than any valid index
    #pragma omp parallel for reduction(min:min_idx) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (a[i] > k) {
            // Update reduction variable; the final reduction will keep the smallest i.
            min_idx = i;
        }
    }

    // Write results.
    if (min_idx < LEN_1D) {
        out_index[0] = min_idx;
        out_value[0] = a[min_idx];
    } else {
        out_index[0] = -1;
        out_value[0] = -1.0;
    }
}

