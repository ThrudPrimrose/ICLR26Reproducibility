#include <stdint.h>

// Reference kernel: versioned_distance_update
// Performs a runtime-distance recurrence: a[i] = 0.75 * a[i-K] + b[i] * c[i]
// The recurrence is applied for i = K .. LEN_1D-1. The first K elements of a are seeds.
// For K <= 0, the kernel does nothing (matching Python semantics where range(K, LEN_1D) is empty).
// Parallelizes across K independent chains using OpenMP.

void versioned_distance_update_fp64(double *restrict a,
                                    const double *restrict b,
                                    const double *restrict c,
                                    const int64_t K,
                                    const int64_t LEN_1D) {
    if (K <= 0) {
        return;
    }
    // Parallelize over the K chains. Each chain updates a[i] where i % K == j.
    #pragma omp parallel for schedule(static) // Parallel over j
    for (int64_t j = 0; j < K; ++j) {
        // Process elements i = j + K, j + 2*K, ... < LEN_1D
        for (int64_t i = j + K; i < LEN_1D; i += K) {
            a[i] = 0.75 * a[i - K] + b[i] * c[i];
        }
    }
}
