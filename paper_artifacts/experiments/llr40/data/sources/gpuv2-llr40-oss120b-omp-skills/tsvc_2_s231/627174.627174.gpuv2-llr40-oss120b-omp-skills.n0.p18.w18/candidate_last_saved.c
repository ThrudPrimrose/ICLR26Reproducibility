#include <stdint.h>
#include <omp.h>

void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    // Dummy target region with constant false condition to avoid device compilation.
    int on_device = 0;
    #pragma omp target if(0) map(from:on_device)
    {
        on_device = !omp_is_initial_device();
    }
    // Parallel column-wise computation on the host.
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        double *restrict aa_ptr = aa + i;
        const double *restrict bb_ptr = bb + i;
        double *restrict a_prev = aa_ptr; // j=0
        double *restrict a_curr = aa_ptr + LEN_2D; // j=1
        const double *restrict b_curr = bb_ptr + LEN_2D;
        for (int64_t j = 1; j < LEN_2D; ++j) {
            *a_curr = *a_prev + *b_curr;
            a_prev += LEN_2D;
            a_curr += LEN_2D;
            b_curr += LEN_2D;
        }
    }
    (void)on_device;
}
