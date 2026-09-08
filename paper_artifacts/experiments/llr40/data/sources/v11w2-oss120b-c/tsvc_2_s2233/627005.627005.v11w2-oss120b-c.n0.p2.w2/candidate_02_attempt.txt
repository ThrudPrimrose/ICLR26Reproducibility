#include <stdint.h>

// Parallel version of tsvc_2_s2233_fp64.
// The first scan (aa) is column‑wise and has a true dependency across rows.
// We parallelise over columns because each column is independent, while the
// inner loop walks the rows with a stride of LEN_2D.
// The second scan (bb) is row‑wise; the inner loop operates on contiguous data
// and is vectorised with OpenMP SIMD.

void tsvc_2_s2233_fp64(double *restrict aa,
                       double *restrict bb,
                       const double *restrict cc,
                       const int64_t LEN_2D) {
    const int64_t START = 8;
    const int64_t N = LEN_2D;
    if (N <= START) return;

    // --------------------------------------------------------------------
    // Prefix‑sum over rows for each column of `aa`.
    // Each column can be processed independently, so we parallelise the outer
    // loop over columns.
    // --------------------------------------------------------------------
    #pragma omp parallel for schedule(static)
    for (int64_t i = START; i < N; ++i) {
        for (int64_t j = START; j < N; ++j) {
            aa[j * N + i] = aa[(j - 1) * N + i] + cc[j * N + i];
        }
    }

    // --------------------------------------------------------------------
    // Prefix‑sum over rows for each row of `bb`.
    // The recurrence is across rows (i) but the inner loop iterates over
    // columns (j) which are contiguous, enabling SIMD vectorisation.
    // --------------------------------------------------------------------
    for (int64_t i = START; i < N; ++i) {
        const double *restrict cc_row = cc + i * N;
        const double *restrict bb_prev = bb + (i - 1) * N;
        double *restrict bb_cur = bb + i * N;
        #pragma omp simd
        for (int64_t j = START; j < N; ++j) {
            bb_cur[j] = bb_prev[j] + cc_row[j];
        }
    }
}

