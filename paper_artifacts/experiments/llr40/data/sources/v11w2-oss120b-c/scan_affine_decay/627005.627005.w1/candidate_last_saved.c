#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

/* Simple sequential implementation of the variable-coefficient affine scan. */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

void __attribute__((optimize("no-tree-vectorize"))) scan_affine_decay_fp64(double * y, double * c, double * x, int64_t LEN_1D, uint8_t * workspace, int64_t workspace_bytes) {
    if (LEN_1D <= 0) {
        return;
    }
    double prev = x[0];
    y[0] = prev;
    for (int64_t i = 1; i < LEN_1D; ++i) {
        double coeff = c[i];
        double cur = coeff * prev + x[i];
        y[i] = cur;
        prev = cur;
    }
}
