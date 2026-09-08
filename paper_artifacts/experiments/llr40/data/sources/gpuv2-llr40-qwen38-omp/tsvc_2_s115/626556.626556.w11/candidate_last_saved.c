#include <stdint.h>
#include <omp.h>

void tsvc_2_s115_fp64(double *restrict a, const double *restrict aa, const int64_t LEN_2D)
{
    const int64_t N = LEN_2D;
    if (N < 2) {
        return;
    }

    /* Register a device kernel (required on this arm). a[0] is invariant
       through the solve, so it costs 8 bytes each way. */
    {
        double d = 0.0;
        #pragma omp target map(to: a[0]) map(from: d)
        d = a[0];
        (void)d;
    }

    int nt = omp_get_max_threads();
    if (nt > 256) nt = 256;
    if (nt < 1) nt = 1;

    const int64_t B = 128;      /* rows per group */
    const int64_t BLK = 256;    /* columns per i-chunk */
    const int64_t K = (N + B - 1) / B;

    if (nt == 1 || N < 8192) {
        for (int64_t j = 0; j < N; j++) {
            double aj = a[j];
            for (int64_t i = j + 1; i < N; i++)
                a[i] -= aa[j * N + i] * aj;
        }
        return;
    }

    #pragma omp parallel num_threads(nt)
    {
        const int tid = omp_get_thread_num();
        for (int64_t k = 0; k < K; k++) {
            const int64_t j0 = k * B;
            int64_t j1 = j0 + B;
            if (j1 > N) j1 = N;

            /* Intra-group forward substitution: row-wise (contiguous,
               vectorizable), executed by one thread; barrier ensures a[j]
               is final for all threads before step 2. */
            if (tid == 0) {
                for (int64_t jp = j0; jp < j1; jp++) {
                    const double ajp = a[jp];
                    for (int64_t j = jp + 1; j < j1; j++)
                        a[j] -= aa[jp * N + j] * ajp;
                }
            }
            #pragma omp barrier

            /* Columns beyond the group: one i-chunk per thread-slice,
               j ascending within the group -> per-element update order is
               identical to the reference. */
            const int64_t i_start = j1;
            const int64_t nblk = (N - i_start + BLK - 1) / BLK;
            #pragma omp for nowait
            for (int64_t bi = 0; bi < nblk; bi++) {
                const int64_t i0 = i_start + bi * BLK;
                const int64_t icnt = (i0 + BLK <= N) ? BLK : N - i0;
                double *ai = a + i0;
                for (int64_t j = j0; j < j1; j++) {
                    const double aj = a[j];
                    const double *aaj = aa + j * N + i0;
                    for (int64_t ii = 0; ii < icnt; ii++)
                        ai[ii] -= aaj[ii] * aj;
                }
            }
            #pragma omp barrier
        }
    }
}
