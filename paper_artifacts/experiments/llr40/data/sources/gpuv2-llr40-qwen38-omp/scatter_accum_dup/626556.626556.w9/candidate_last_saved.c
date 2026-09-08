#include <stdint.h>

void scatter_accum_dup_fp64(double *bins, const int32_t *ip, const double *src,
                            const int64_t LEN_1D, uint8_t *ws, const int64_t wsl) {
    (void)ws; (void)wsl;
    if (LEN_1D <= 0) return;
    #pragma omp target teams distribute parallel for \
        map(tofrom: bins[0:LEN_1D]) map(to: src[0:LEN_1D]) map(to: ip[0:LEN_1D])
    for (int64_t i = 0; i < LEN_1D; i++) {
        #pragma omp atomic update
        bins[ip[i]] += src[i];
    }
}
