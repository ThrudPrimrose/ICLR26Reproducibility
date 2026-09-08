#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void tsvc_2_s233_fp64(double *restrict aa, double *restrict bb,
                      const double *restrict cc, const int64_t LEN_2D) {
    const int64_t N = LEN_2D;
    const int64_t M = N - 8;           /* submatrix side */
    const int64_t N2 = N * N;

    /* persistent host scratch (allocated once per M) */
    static double *aa_out = NULL, *bb_out = NULL, *aa_seed = NULL, *bb_seed = NULL;
    static int64_t cur_M = -1;
    if (cur_M != M) {
        free(aa_out); free(bb_out); free(aa_seed); free(bb_seed);
        aa_out  = malloc((size_t)M * M * sizeof(double));
        bb_out  = malloc((size_t)M * M * sizeof(double));
        aa_seed = malloc((size_t)M * sizeof(double));
        bb_seed = malloc((size_t)M * sizeof(double));
        cur_M = M;
    }

    /* gather seeds on the host (small) */
    memcpy(aa_seed, aa + 7 * N + 8, (size_t)M * sizeof(double));   /* row 7, contiguous */
    #pragma omp parallel for schedule(static)
    for (int64_t j = 8; j < N; ++j) bb_seed[j - 8] = bb[j * N + 7]; /* col 7, strided */

    #pragma omp target data map(to: cc[0:N2]) map(to: aa_seed[0:M]) map(to: bb_seed[0:M]) \
                            map(from: aa_out[0:(size_t)M*M]) map(from: bb_out[0:(size_t)M*M])
    {
        /* aa: parallel over column i, serial scan over row j */
        #pragma omp target teams distribute parallel for
        for (int64_t i = 8; i < N; ++i) {
            double acc = aa_seed[i - 8];
            for (int64_t j = 8; j < N; ++j) {
                acc += cc[j * N + i];
                aa_out[(j - 8) * M + (i - 8)] = acc;
            }
        }
        /* bb: parallel over row j, serial scan over column i */
        #pragma omp target teams distribute parallel for
        for (int64_t j = 8; j < N; ++j) {
            double acc = bb_seed[j - 8];
            for (int64_t i = 8; i < N; ++i) {
                acc += cc[j * N + i];
                bb_out[(j - 8) * M + (i - 8)] = acc;
            }
        }
    }

    /* scatter submatrices back into the host aa/bb (row-wise contiguous copies) */
    #pragma omp parallel for schedule(static)
    for (int64_t j = 8; j < N; ++j)
        memcpy(aa + j * N + 8, aa_out + (j - 8) * M, (size_t)M * sizeof(double));
    #pragma omp parallel for schedule(static)
    for (int64_t j = 8; j < N; ++j)
        memcpy(bb + j * N + 8, bb_out + (j - 8) * M, (size_t)M * sizeof(double));
}
