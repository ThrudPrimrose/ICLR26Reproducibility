#include <stdint.h>
#include <omp.h>

void scan_affine_decay_fp64(double *restrict y,
                            double *restrict c,
                            double *restrict x,
                            int64_t LEN_1D,
                            uint8_t *restrict workspace,
                            int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;

    int64_t n = LEN_1D;
    if (n <= 0) return;
    y[0] = x[0];
    if (n <= 1) return;

    int64_t N1 = n - 1;
    int nt = omp_get_max_threads();

    // For very small problems stay serial to avoid parallel overhead.
    if (N1 < 256 || nt <= 1) {
        for (int64_t i = 1; i < n; ++i)
            y[i] = c[i] * y[i - 1] + x[i];
        return;
    }

    // One contiguous block per thread for the remaining N1 elements.
    int64_t block = (N1 + nt - 1) / nt;
    int M = (int)((N1 + block - 1) / block);

    // Per-block scratch: a (product), s (local tail), L (left input for pass 3).
    double *scratch = (double *)__builtin_alloca(3 * M * sizeof(double));
    double *a = scratch;
    double *s = scratch + M;
    double *L = scratch + 2 * M;

    // Pass 1: each thread scans its block assuming a zero left input and records
    // the block's total product (a) and the local tail value (s).
    #pragma omp parallel for schedule(static) default(none) \
        shared(y, c, x, a, s, n, block, M)
    for (int b = 0; b < M; ++b) {
        int64_t start = 1 + (int64_t)b * block;
        int64_t end = start + block;
        if (end > n) end = n;

        y[start] = x[start];
        double prod = c[start];
        for (int64_t i = start + 1; i < end; ++i) {
            y[i] = c[i] * y[i - 1] + x[i];
            prod *= c[i];
        }
        a[b] = prod;
        s[b] = y[end - 1];
    }

    // Pass 2: serial prefix across block aggregates gives the true left input for
    // every block.  The product path is associative and contracts, so this
    // reassociation stays inside the floating-point band of the serial oracle.
    double left = y[0];
    for (int b = 0; b < M; ++b) {
        L[b] = left;
        left = a[b] * left + s[b];
    }

    // Pass 3: recompute each block with its true left input.
    #pragma omp parallel for schedule(static) default(none) \
        shared(y, c, x, L, n, block, M)
    for (int b = 0; b < M; ++b) {
        int64_t start = 1 + (int64_t)b * block;
        int64_t end = start + block;
        if (end > n) end = n;

        double prev = c[start] * L[b] + x[start];
        y[start] = prev;
        for (int64_t i = start + 1; i < end; ++i) {
            prev = c[i] * prev + x[i];
            y[i] = prev;
        }
    }
}
