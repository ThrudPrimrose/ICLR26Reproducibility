#include <stdint.h>

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
    const int64_t N = LEN_2D;
    if (N < 9) return;
    const int64_t ncol = N - 8;
    const int64_t n4 = (ncol / 4) * 4;

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int64_t c = 8; c < 8 + n4; c += 4) {
            double da0 = aa[7 * N + c];
            double da1 = aa[7 * N + c + 1];
            double da2 = aa[7 * N + c + 2];
            double da3 = aa[7 * N + c + 3];
            double db0 = bb[7 * N + c];
            double db1 = bb[7 * N + c + 1];
            double db2 = bb[7 * N + c + 2];
            double db3 = bb[7 * N + c + 3];
            int64_t idx = 8 * N + c;
            for (int64_t j = 8; j < N; ++j) {
                const double t0 = cc[idx];
                const double t1 = cc[idx + 1];
                const double t2 = cc[idx + 2];
                const double t3 = cc[idx + 3];
                da0 += t0; db0 += t0; aa[idx] = da0;   bb[idx] = db0;
                da1 += t1; db1 += t1; aa[idx + 1] = da1; bb[idx + 1] = db1;
                da2 += t2; db2 += t2; aa[idx + 2] = da2; bb[idx + 2] = db2;
                da3 += t3; db3 += t3; aa[idx + 3] = da3; bb[idx + 3] = db3;
                idx += N;
            }
        }
        #pragma omp for schedule(static)
        for (int64_t c = 8 + n4; c < N; ++c) {
            double da = aa[7 * N + c];
            double db = bb[7 * N + c];
            int64_t idx = 8 * N + c;
            for (int64_t j = 8; j < N; ++j) {
                const double t = cc[idx];
                da += t;
                db += t;
                aa[idx] = da;
                bb[idx] = db;
                idx += N;
            }
        }
    }
}
