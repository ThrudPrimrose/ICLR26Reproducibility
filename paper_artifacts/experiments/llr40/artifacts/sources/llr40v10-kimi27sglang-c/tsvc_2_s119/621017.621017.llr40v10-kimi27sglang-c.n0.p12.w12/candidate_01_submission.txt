#include <stdint.h>
#include <omp.h>

static void __attribute__((noinline)) s119_serial(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    for (int64_t i = 1; i < LEN_2D; ++i) {
        for (int64_t j = 1; j < LEN_2D; ++j) {
            const int64_t idx_ij = i * LEN_2D + j;
            const int64_t idx_im1j = (i - 1) * LEN_2D + (j - 1);
            aa[idx_ij] = aa[idx_im1j] + bb[idx_ij];
        }
    }
}

static void __attribute__((noinline)) s119_parallel(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    const int64_t stride = LEN_2D + 1;
    const int64_t s_min = -(LEN_2D - 2);
    const int64_t s_max = LEN_2D - 2;

    #pragma omp parallel for schedule(dynamic, 64)
    for (int64_t s = s_min; s <= s_max; ++s) {
        int64_t i0, i1;
        if (s >= 0) {
            i0 = 1;
            i1 = LEN_2D - 1 - s;
        } else {
            i0 = 1 - s;
            i1 = LEN_2D - 1;
        }

        int64_t idx = i0 * stride + s;
        int64_t pidx = (i0 - 1) * stride + s;
        int64_t bidx = i0 * LEN_2D + (i0 + s);

        for (int64_t i = i0; i <= i1; ++i) {
            aa[idx] = aa[pidx] + bb[bidx];
            idx += stride;
            pidx += stride;
            bidx += stride;
        }
    }
}

void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    if (LEN_2D < 2048) {
        s119_serial(aa, bb, LEN_2D);
    } else {
        s119_parallel(aa, bb, LEN_2D);
    }
}
