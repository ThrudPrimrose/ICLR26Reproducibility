#include <stdint.h>

void fdtd_2d_fp64(double *restrict ex, double *restrict ey, const double *restrict fict, double *restrict hz, const int64_t NX, const int64_t NY, const int64_t TMAX, const double ex_courant, const double ey_courant, const double hz_courant) {
    if (NX <= 0 || NY <= 0 || TMAX <= 0) return;

    for (int64_t t = 0; t < TMAX; ++t) {
        /* ey[0, :] = fict[t] */
        for (int64_t j = 0; j < NY; ++j) {
            ey[j] = fict[t];
        }

        /* Pipeline rows: update ey[i+1], ex[i], hz[i] in order.
         * This preserves the reference's ey -> ex -> hz update order within
         * each row while keeping the working set cache-resident. */
        for (int64_t i = 0; i < NX - 1; ++i) {
            double *ey_next = ey + (i + 1) * NY;
            const double *hz_next = hz + (i + 1) * NY;
            const double *hz_cur  = hz + i * NY;

            /* ey[i+1, :] -= ey_courant * (hz[i+1, :] - hz[i, :]) */
            for (int64_t j = 0; j < NY; ++j) {
                ey_next[j] -= ey_courant * (hz_next[j] - hz_cur[j]);
            }

            double *ex_cur = ex + i * NY;
            /* ex[i, 1:] -= ex_courant * (hz[i, 1:] - hz[i, :-1]) */
            for (int64_t j = 1; j < NY; ++j) {
                ex_cur[j] -= ex_courant * (hz_cur[j] - hz_cur[j - 1]);
            }

            double *hz_rw = hz + i * NY;
            const double *ex_row = ex + i * NY;
            const double *ey_cur = ey + i * NY;
            /* hz[i, :-1] -= hz_courant * ((ex[i, 1:] - ex[i, :-1])
             *                              + ey[i+1, :-1] - ey[i, :-1]) */
            for (int64_t j = 0; j < NY - 1; ++j) {
                hz_rw[j] -= hz_courant * ((ex_row[j + 1] - ex_row[j]) + ey_next[j] - ey_cur[j]);
            }
        }

        /* ex[NX-1, 1:] update (no hz row NX-1 is written) */
        double *ex_last = ex + (NX - 1) * NY;
        const double *hz_last = hz + (NX - 1) * NY;
        for (int64_t j = 1; j < NY; ++j) {
            ex_last[j] -= ex_courant * (hz_last[j] - hz_last[j - 1]);
        }
    }
}
