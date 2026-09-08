/* Debug kernel to print pointer addresses */
#include <stdint.h>
#include <stdio.h>

void segment_reduce_ragged_fp64(const int64_t *restrict row_ptr,
                                const double *restrict val,
                                const double *restrict w,
                                double *restrict out,
                                const int64_t arg) {
    printf("Arg=%lld\n", (long long)arg);
    printf("row_ptr=%p val=%p w=%p out=%p\n", (void*)row_ptr, (void*)val, (void*)w, (void*)out);
    fflush(stdout);
    // Print first element of each as double (or int64 for row_ptr)
    printf("row_ptr[0]=%lld\n", (long long)row_ptr[0]);
    printf("val[0]=%f\n", val[0]);
    printf("w[0]=%f\n", w[0]);
    printf("out[0]=%f\n", out[0]);
    fflush(stdout);
}
