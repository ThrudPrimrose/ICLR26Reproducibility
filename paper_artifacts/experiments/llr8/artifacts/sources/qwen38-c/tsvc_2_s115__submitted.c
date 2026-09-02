#include <stdint.h>
#include <omp.h>

static void s115_serial(double *__restrict ar, const double *__restrict aar, int64_t N)
{
    for (int64_t j = 0; j < N; j++) {
        const double aj = ar[j];
        for (int64_t i = j + 1; i < N; i++)
            ar[i] = ar[i] - aar[j * N + i] * aj;
    }
}

void tsvc_2_s115_fp64(double *a, double *aa, int64_t LEN_2D, uint8_t *workspace,
                      int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;
    const int64_t N = LEN_2D;
    if (N < 2)
        return;

    const int64_t B = 256;
    const int64_t nb = (N + B - 1) / B;
    double *__restrict ar = a;
    const double *__restrict aar = aa;

    if (nb < 8) {
        s115_serial(ar, aar, N);
        return;
    }

    #pragma omp parallel
    {
        for (int64_t b = 0; b < nb; b++) {
            const int64_t j0 = b * B;
            const int64_t j1 = j0 + B < N ? j0 + B : N;
            #pragma omp single
            {
                for (int64_t j = j0; j < j1; j++) {
                    const double aj = ar[j];
                    for (int64_t i = j + 1; i < j1; i++)
                        ar[i] = ar[i] - aar[j * N + i] * aj;
                }
            }
            #pragma omp for schedule(static)
            for (int64_t d = b + 1; d < nb; d++) {
                const int64_t i0 = d * B;
                const int64_t i1 = i0 + B < N ? i0 + B : N;
                for (int64_t j = j0; j < j1; j++) {
                    const double aj = ar[j];
                    const double *row = aar + j * N;
                    for (int64_t i = i0; i < i1; i++)
                        ar[i] = ar[i] - row[i] * aj;
                }
            }
        }
    }
}
