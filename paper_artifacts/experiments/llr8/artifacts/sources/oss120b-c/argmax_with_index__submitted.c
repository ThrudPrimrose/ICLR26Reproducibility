#include <stddef.h>
#include <stdint.h>

// Compute the maximum value in the input array and its index.
// The function follows the reference Python semantics: for ties, the first index is chosen.
// Parameters:
//   a         - input array of length LEN_1D (double precision)
//   out_value - pointer to a single-element array to store the maximum value
//   out_index - pointer to a single-element array to store the index (int64_t) of the maximum value
//   LEN_1D    - number of elements in the input array
// All pointers are assumed to be non-null and properly aligned.

void argmax_with_index_fp64(const double * __restrict a,
                        int64_t * __restrict out_index,
                        double * __restrict out_value,
                        int64_t LEN_1D,
                        uint8_t *workspace,
                        int64_t workspace_len) {
    if (LEN_1D <= 0) {
        // Undefined behavior in benchmark; set defaults.
        if (out_value) *out_value = 0.0;
        if (out_index) *out_index = -1;
        return;
    }

    // First pass: vectorized reduction to find the maximum value.
    double max_val = a[0];
    #pragma omp simd reduction(max:max_val)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double v = a[i];
        if (v > max_val) max_val = v;
    }

    // Second pass: find the first index where the maximum occurs.
    int64_t idx = 0;
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (a[i] == max_val) {
            idx = i;
            break;
        }
    }

    *out_value = max_val;
    *out_index = idx;
    (void)workspace;
    (void)workspace_len;
}
