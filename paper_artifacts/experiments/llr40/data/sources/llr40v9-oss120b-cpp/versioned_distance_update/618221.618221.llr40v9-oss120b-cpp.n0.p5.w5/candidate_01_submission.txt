/*
 * versioned_distance_update kernel – C++ implementation.
 *
 * The reference NumPy implementation (see /shared/tasks/versioned_distance_update/versioned_distance_update_numpy.py)
 * defines the recurrence:
 *   a[i] = 0.75 * a[i - K] + b[i] * c[i]
 * for i in [K, LEN_1D).
 *
 * The kernel is called with double‑precision arrays. The benchmark driver expects a C ABI
 * symbol named <kernel>_fp64 where <kernel> is "versioned_distance_update". The signature matches
 * the other reference kernels in the suite (e.g. argmax_with_index_fp64):
 *
 *   void versioned_distance_update_fp64(double *__restrict__ a,
 *                                       const double *__restrict__ b,
 *                                       const double *__restrict__ c,
 *                                       const int64_t K,
 *                                       const int64_t K);
 *
 * The implementation parallelises the K independent chains with OpenMP. Each chain corresponds
 * to a fixed offset d (0 <= d < K) and can be processed sequentially because the recurrence only
 * depends on the previous element in the same chain. This yields perfect scalability for large K
 * while preserving the exact semantics required for K = 1.
 */

#include <cstddef>
#include <cstdint>

extern "C" void versioned_distance_update_fp64(double *__restrict__ a,
                                                const double *__restrict__ b,
                                                const double *__restrict__ c,
                                                const int64_t K,
                                                const int64_t LEN_1D) {
    // Guard against degenerate parameters. The benchmark guarantees K > 0 and LEN_1D >= K,
    // but we defensively handle any other case to avoid undefined behaviour.
    if (K <= 0 || LEN_1D <= K) {
        // Nothing to do – the loop range would be empty.
        return;
    }

    // Parallelise over the K independent recurrence chains.
    // Each chain processes indices i = d + K, d + 2*K, ..., < LEN_1D.
    // The outer loop iterates over the starting offset d.
#pragma omp parallel for schedule(static)
    for (int64_t d = 0; d < K; ++d) {
        // Compute the first index in this chain that requires an update.
        // The element at index d (if d < K) is a seed and must not be overwritten.
        // The first writable element is d + K.
        for (int64_t i = d + K; i < LEN_1D; i += K) {
            a[i] = 0.75 * a[i - K] + b[i] * c[i];
        }
    }
}

