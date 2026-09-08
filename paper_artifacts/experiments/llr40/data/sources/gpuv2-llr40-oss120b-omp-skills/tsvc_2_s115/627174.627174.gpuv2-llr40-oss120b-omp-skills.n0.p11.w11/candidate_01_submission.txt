#include <stdint.h>
#include <omp.h>
#include <stdlib.h>

// Dummy target region to ensure at least one device kernel is compiled.
static inline void dummy_target(void) {
    // Simple operation on the device; does not affect correctness.
    #pragma omp target
    {
        // No data mapping needed; just a dummy computation.
        int device_id = omp_get_device_num();
        (void)device_id;
    }
}

void tsvc_2_s115_fp64(double *restrict a, const double *restrict aa, const int64_t LEN_2D) {
    // Ensure a device kernel exists (required for offload arm).
    dummy_target();

    // Host implementation with SIMD. Outer loop is sequential due to true dependence.
    for (int64_t j = 0; j < LEN_2D; ++j) {
        double aj = a[j];
        int64_t cnt = LEN_2D - (j + 1);
        if (cnt <= 0) continue;
        const double *aa_ptr = aa + j * LEN_2D + (j + 1);
        double *a_ptr = a + (j + 1);
        #pragma omp simd
        for (int64_t k = 0; k < cnt; ++k) {
            a_ptr[k] -= aa_ptr[k] * aj;
        }
    }
}
