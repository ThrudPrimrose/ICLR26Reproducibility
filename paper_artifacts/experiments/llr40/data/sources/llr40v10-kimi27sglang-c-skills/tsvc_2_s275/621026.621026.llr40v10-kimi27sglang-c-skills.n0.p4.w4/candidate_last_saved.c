#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

static inline void *aligned_size(size_t bytes) {
    size_t sz = (bytes + 63) & ~((size_t)63);
    return aligned_alloc(64, sz);
}

void tsvc_2_s275_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
    const int64_t n = LEN_2D;

    uint32_t *restrict act = aligned_size((size_t)n * sizeof(uint32_t));
    double *restrict init = aligned_size((size_t)n * sizeof(double));
    int64_t na = 0;
    for (int64_t i = 0; i < n; i++) {
        if (aa[i] > 0.0) {
            act[na] = (uint32_t)i;
            init[na] = aa[i];
            na++;
        }
    }

    if (na == 0) {
        free(act);
        free(init);
        return;
    }

    if (na * 20 < n) {
        #pragma omp parallel for schedule(static)
        for (int64_t k = 0; k < na; k++) {
            const int64_t i = (int64_t)act[k];
            double prev = init[k];
            for (int64_t j = 1; j < n; j++) {
                const int64_t idx = j * n + i;
                prev = prev + bb[idx] * cc[idx];
                aa[idx] = prev;
            }
        }
    } else {
        #pragma omp parallel
        {
            const int nt = omp_get_num_threads();
            const int tid = omp_get_thread_num();
            const int64_t a0 = (na * tid) / nt;
            const int64_t a1 = (na * (tid + 1)) / nt;
            const int64_t cnt = a1 - a0;
            if (cnt > 0) {
                double prev[cnt];
                for (int64_t k = 0; k < cnt; k++) {
                    prev[k] = init[a0 + k];
                }
                const double *b = bb;
                const double *c = cc;
                double *a = aa;
                for (int64_t j = 1; j < n; j++) {
                    b += n;
                    c += n;
                    a += n;
                    #pragma GCC unroll 8
                    for (int64_t k = 0; k < cnt; k++) {
                        const int64_t i = (int64_t)act[a0 + k];
                        const double nv = prev[k] + b[i] * c[i];
                        a[i] = nv;
                        prev[k] = nv;
                    }
                }
            }
        }
    }

    free(act);
    free(init);
}
