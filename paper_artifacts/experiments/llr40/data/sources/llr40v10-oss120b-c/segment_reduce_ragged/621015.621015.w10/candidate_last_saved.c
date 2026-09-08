#include <stdint.h>
#include <stdio.h>

#pragma GCC push_options
#pragma GCC optimize ("O0")
void segment_reduce_ragged_fp64(const double *val,
                                const int64_t *row_ptr,
                                const double *w,
                                double *out,
                                int64_t NSEG,
                                uint8_t *workspace,
                                int64_t workspace_bytes) {
    for (int64_t s = 0; s < NSEG; ++s) {
        // Debug print for first few segments
        if (s < 5) {
            fprintf(stderr, "segment %ld start=%ld end=%ld\n", s, row_ptr[s], row_ptr[s+1]);
            fflush(stderr);
        }
        double acc = 0.0;
        int64_t start = row_ptr[s];
        int64_t end = row_ptr[s + 1];
        if (s == 5) {
            fprintf(stderr, "DEBUG start5=%ld end5=%ld\n", start, end);
            fflush(stderr);
        }
        if (s == 6) {
            fprintf(stderr, "DEBUG start6=%ld end6=%ld\n", start, end);
            fflush(stderr);
        }
        if (s == 7) {
            fprintf(stderr, "DEBUG start7=%ld end7=%ld\n", start, end);
            fflush(stderr);
        }
        if (s == 8) {
            fprintf(stderr, "DEBUG start8=%ld end8=%ld\n", start, end);
            fflush(stderr);
        }
        for (int64_t e = start; e < end; ++e) {
            acc += val[e] * w[e];
        }
        out[s] = acc;
    }
    (void)workspace;
    (void)workspace_bytes;
}
#pragma GCC pop_options
