#include <stdint.h>
#include <stddef.h>
#include <math.h>

typedef struct { double maxv; int64_t idx; } maxidx_t;

static void combine(maxidx_t *a, maxidx_t *b) {
    if (b->maxv > a->maxv || (b->maxv == a->maxv && b->idx < a->idx)) {
        a->maxv = b->maxv;
        a->idx = b->idx;
    }
}

#pragma omp declare reduction(maxidx:maxidx_t:combine(&omp_out, &omp_in)) initializer(omp_priv={-INFINITY,INT64_MAX})

void tsvc_2_s13110_fp64(double *aa, double *bb, int64_t LEN_2D, uint8_t *workspace, int64_t workspace_bytes) {
    int64_t n = LEN_2D * LEN_2D;
    maxidx_t mx = {-INFINITY, INT64_MAX};
    #pragma omp parallel for reduction(maxidx:mx) schedule(static)
    for (int64_t k = 0; k < n; ++k) {
        double v = aa[k];
        if (v > mx.maxv) {
            mx.maxv = v;
            mx.idx = k;
        }
    }
    int64_t xindex = mx.idx / LEN_2D;
    int64_t yindex = mx.idx % LEN_2D;
    bb[0] = mx.maxv + (double)xindex + (double)yindex;
}
