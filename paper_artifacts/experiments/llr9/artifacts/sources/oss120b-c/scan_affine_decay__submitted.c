#include <omp.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct {
    double A; // multiplicative coefficient
    double B; // additive coefficient
} Transform;

// Scan affine decay: y[i] = c[i] * y[i-1] + x[i]
// y, c, x are arrays of length LEN_1D. y[0] is set to x[0] (seed).
void scan_affine_decay_fp64(double *restrict y_arg, const double *restrict c_arg, const double *restrict x_arg, int64_t LEN_1D, uint8_t *restrict workspace, int64_t workspace_bytes) {
    // Argument order in harness: first argument is coefficients (c), second is input values (x), third is output y.
    const double *coeff = y_arg; // coefficients
    const double *input = c_arg; // input values
    double *out = (double *)x_arg; // output array

    if (LEN_1D <= 0) return;
    // Seed: initialize first output element.
    out[0] = input[0];
    if (LEN_1D == 1) return;

    // Determine number of threads to use; fallback to serial for tiny problems.
    int max_threads = omp_get_max_threads();
    if (LEN_1D < 4 * max_threads) {
        // Small case: just run serial loop.
        for (int64_t i = 1; i < LEN_1D; ++i) {
            out[i] = coeff[i] * out[i - 1] + input[i];
        }
        return;
    }

    // Allocate per‑thread block transforms.
    Transform *block_t = (Transform *)malloc(max_threads * sizeof(Transform));
    Transform *prefix_t = (Transform *)malloc(max_threads * sizeof(Transform));
    if (!block_t || !prefix_t) {
        // Allocation failed – fall back to serial.
        for (int64_t i = 1; i < LEN_1D; ++i) {
            out[i] = coeff[i] * out[i - 1] + input[i];
        }
        if (block_t) free(block_t);
        if (prefix_t) free(prefix_t);
        return;
    }
    // Initialize block transforms to identity (handles empty blocks).
    for (int t = 0; t < max_threads; ++t) {
        block_t[t].A = 1.0;
        block_t[t].B = 0.0;
    }

    // ------- First pass: compute block transforms (A,B) -------
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        // Chunk size for this thread (ceil division).
        int64_t chunk = (LEN_1D + nt - 1) / nt;
        int64_t start = (int64_t)tid * chunk;
        int64_t end = start + chunk;
        if (end > LEN_1D) end = LEN_1D;

        Transform tr;
        tr.A = 1.0;
        tr.B = 0.0;
        // The recurrence starts at i = 1, so skip i = 0.
        int64_t i_start = (start < 1) ? 1 : start;
        for (int64_t i = i_start; i < end; ++i) {
            tr.B = coeff[i] * tr.B + input[i];
            tr.A *= coeff[i];
        }
        block_t[tid] = tr;
    } // implicit barrier

    // ------- Compute prefix transforms of blocks (sequential) -------
    prefix_t[0].A = 1.0;   // identity
    prefix_t[0].B = 0.0;
    for (int t = 1; t < max_threads; ++t) {
        Transform prev = prefix_t[t - 1];
        Transform blk = block_t[t - 1];
        Transform cur;
        cur.A = blk.A * prev.A;
        cur.B = blk.A * prev.B + blk.B;
        prefix_t[t] = cur;
    }

    // ------- Second pass: write out using the proper initial value for each block -------
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        int64_t chunk = (LEN_1D + nt - 1) / nt;
        int64_t start = (int64_t)tid * chunk;
        int64_t end = start + chunk;
        if (end > LEN_1D) end = LEN_1D;

        double y_before;
        if (start == 0) {
            // First block: out[0] already set.
            y_before = out[0];
            for (int64_t i = 1; i < end; ++i) {
                out[i] = coeff[i] * y_before + input[i];
                y_before = out[i];
            }
        } else {
            // Compute y_before = value of out at index start-1 using prefix transform.
            Transform pre = prefix_t[tid];
            double y0 = out[0];
            y_before = pre.A * y0 + pre.B;
            for (int64_t i = start; i < end; ++i) {
                out[i] = coeff[i] * y_before + input[i];
                y_before = out[i];
            }
        }
    }

    free(block_t);
    free(prefix_t);
}

