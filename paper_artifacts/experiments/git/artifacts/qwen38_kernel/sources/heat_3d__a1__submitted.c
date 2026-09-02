#include <stdint.h>
static void body_s(const double *restrict rowm, const double *restrict rowp, const double *restrict cm, const double *restrict cc, const double *restrict cp, double *restrict d, int64_t k, double alpha) {
    double Ac0 = cc[k+0], A2c0 = 2.0*Ac0;
    double Ac1 = cc[k+8], A2c1 = 2.0*Ac1;
    double Ac2 = cc[k+16], A2c2 = 2.0*Ac2;
    d[k] = alpha * (rowp[k] - A2c0 + rowm[k]) + alpha * (cp[k] - A2c0 + cm[k]) + alpha * (cc[k+1] - A2c0 + cc[k-1]) + Ac0;
    d[k+8] = alpha * (rowp[k+8] - A2c1 + rowm[k+8]) + alpha * (cp[k+8] - A2c1 + cm[k+8]) + alpha * (cc[k+9] - A2c1 + cc[k+7]) + Ac1;
    d[k+16] = alpha * (rowp[k+16] - A2c2 + rowm[k+16]) + alpha * (cp[k+16] - A2c2 + cm[k+16]) + alpha * (cc[k+17] - A2c2 + cc[k+15]) + Ac2;
}
static void do_rows_s(const double *restrict rowm, const double *restrict rowp, const double *restrict cm, const double *restrict cc, const double *restrict cp, double *restrict d, int64_t N, double alpha) {
    int64_t ke = N - 1, k;
    for (k = 1; k + 24 <= ke; k += 24)
        body_s(rowm, rowp, cm, cc, cp, d, k, alpha);
    for (; k < ke; ++k) {
        double Ac = cc[k];
        d[k] = alpha * (rowp[k] - 2.0*Ac + rowm[k]) + alpha * (cp[k] - 2.0*Ac + cm[k]) + alpha * (cc[k+1] - 2.0*Ac + cc[k-1]) + Ac;
    }
}

#include <stdint.h>
static void body_l(const double *restrict rowm, const double *restrict rowp, const double *restrict cm, const double *restrict cc, const double *restrict cp, double *restrict d, int64_t k, double alpha) {
    double Ac0 = cc[k+0], A2c0 = 2.0*Ac0;
    double Ac1 = cc[k+8], A2c1 = 2.0*Ac1;
    double Ac2 = cc[k+16], A2c2 = 2.0*Ac2;
    double Ac3 = cc[k+24], A2c3 = 2.0*Ac3;
    double Ac4 = cc[k+32], A2c4 = 2.0*Ac4;
    double Ac5 = cc[k+40], A2c5 = 2.0*Ac5;
    d[k] = alpha * (rowp[k] - A2c0 + rowm[k]) + alpha * (cp[k] - A2c0 + cm[k]) + alpha * (cc[k+1] - A2c0 + cc[k-1]) + Ac0;
    d[k+8] = alpha * (rowp[k+8] - A2c1 + rowm[k+8]) + alpha * (cp[k+8] - A2c1 + cm[k+8]) + alpha * (cc[k+9] - A2c1 + cc[k+7]) + Ac1;
    d[k+16] = alpha * (rowp[k+16] - A2c2 + rowm[k+16]) + alpha * (cp[k+16] - A2c2 + cm[k+16]) + alpha * (cc[k+17] - A2c2 + cc[k+15]) + Ac2;
    d[k+24] = alpha * (rowp[k+24] - A2c3 + rowm[k+24]) + alpha * (cp[k+24] - A2c3 + cm[k+24]) + alpha * (cc[k+25] - A2c3 + cc[k+23]) + Ac3;
    d[k+32] = alpha * (rowp[k+32] - A2c4 + rowm[k+32]) + alpha * (cp[k+32] - A2c4 + cm[k+32]) + alpha * (cc[k+33] - A2c4 + cc[k+31]) + Ac4;
    d[k+40] = alpha * (rowp[k+40] - A2c5 + rowm[k+40]) + alpha * (cp[k+40] - A2c5 + cm[k+40]) + alpha * (cc[k+41] - A2c5 + cc[k+39]) + Ac5;
}
static void do_rows_l(const double *restrict rowm, const double *restrict rowp, const double *restrict cm, const double *restrict cc, const double *restrict cp, double *restrict d, int64_t N, double alpha) {
    int64_t ke = N - 1, k;
    for (k = 1; k + 48 <= ke; k += 48)
        body_l(rowm, rowp, cm, cc, cp, d, k, alpha);
    for (; k < ke; ++k) {
        double Ac = cc[k];
        d[k] = alpha * (rowp[k] - 2.0*Ac + rowm[k]) + alpha * (cp[k] - 2.0*Ac + cm[k]) + alpha * (cc[k+1] - 2.0*Ac + cc[k-1]) + Ac;
    }
}


void heat_3d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS, double alpha) {
    const int64_t NI = N - 2;
    const int64_t NN = N * N;
    const int64_t NP = NI * NI;
    if (NP <= 0) return;
    const int large = N > 300;
    #pragma omp parallel
    {
        for (int64_t t = 1; t <= TSTEPS; ++t) {
            #pragma omp for schedule(static)
            for (int64_t p = 0; p < NP; ++p) {
                int64_t i = 1 + p / NI, j = 1 + p % NI;
                if (large)
                    do_rows_l(A + (i-1)*NN + j*N, A + (i+1)*NN + j*N, A + i*NN + (j-1)*N, A + i*NN + j*N, A + i*NN + (j+1)*N, B + i*NN + j*N, N, alpha);
                else
                    do_rows_s(A + (i-1)*NN + j*N, A + (i+1)*NN + j*N, A + i*NN + (j-1)*N, A + i*NN + j*N, A + i*NN + (j+1)*N, B + i*NN + j*N, N, alpha);
            }
            #pragma omp for schedule(static)
            for (int64_t p = 0; p < NP; ++p) {
                int64_t i = 1 + p / NI, j = 1 + p % NI;
                if (large)
                    do_rows_l(B + (i-1)*NN + j*N, B + (i+1)*NN + j*N, B + i*NN + (j-1)*N, B + i*NN + j*N, B + i*NN + (j+1)*N, A + i*NN + j*N, N, alpha);
                else
                    do_rows_s(B + (i-1)*NN + j*N, B + (i+1)*NN + j*N, B + i*NN + (j-1)*N, B + i*NN + j*N, B + i*NN + (j+1)*N, A + i*NN + j*N, N, alpha);
            }
        }
    }
}
