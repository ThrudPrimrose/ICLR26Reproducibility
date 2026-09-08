#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define BLK 512   /* elements per first-level block (one work-item each) */
#define G2  2048  /* first-level blocks per second-level group */

void scan_affine_decay_fp64(double *c, double *x, double *y, int64_t n,
                            uint8_t *ws, int64_t ws_size)
{
    (void)ws;
    (void)ws_size;
    if (n <= 0) return;
    if (n == 1) { y[0] = x[0]; return; }

    const int64_t nb  = (n + BLK - 1) / BLK;
    const int64_t nb2 = (nb + G2 - 1) / G2;
    const int64_t need = 2 * nb + 2 * nb2 + nb2;

    static double *scratch = NULL;
    static int64_t scratch_n = 0;
    if (scratch_n < need) {
        free(scratch);
        scratch = (double *)malloc((size_t)need * sizeof(double));
        scratch_n = need;
    }
    double *pair  = scratch;            /* 2*nb doubles */
    double *gpair = scratch + 2 * nb;   /* 2*nb2 doubles */
    double *rc    = scratch + 2 * nb + 2 * nb2; /* nb2 doubles */

    #pragma omp target data map(to: c[0:n]) map(to: x[0:n]) map(from: y[0:n]) \
            map(to: scratch[0:need])
    {
        /* pass 1: per-block local scan -> (A_k, B_k) */
        #pragma omp target teams distribute parallel for
        for (int64_t k = 0; k < nb; k++) {
            const int64_t s = k * BLK;
            int64_t e = s + BLK;
            if (e > n) e = n;
            double acc = 0.0, A = 1.0;
            for (int64_t i = s; i < e; i++) {
                acc = c[i] * acc + x[i];
                A *= c[i];
            }
            pair[2 * k] = A;
            pair[2 * k + 1] = acc;
        }

        /* pass 2: per-group prefix of block pairs + group map */
        #pragma omp target teams distribute parallel for
        for (int64_t j = 0; j < nb2; j++) {
            int64_t k0 = j * G2;
            int64_t k1 = k0 + G2;
            if (k1 > nb) k1 = nb;
            double p = 1.0, q = 0.0;
            for (int64_t k = k0; k < k1; k++) {
                const double Ak = pair[2 * k];
                const double Bk = pair[2 * k + 1];
                pair[2 * k] = p;
                pair[2 * k + 1] = q;
                q = Ak * q + Bk;
                p = Ak * p;
            }
            gpair[2 * j] = p;
            gpair[2 * j + 1] = q;
        }

        /* serial scan over the few remaining groups, one work-item */
        #pragma omp target teams num_teams(1)
        {
            double r = 0.0;
            for (int64_t j = 0; j < nb2; j++) {
                rc[j] = r;
                r = gpair[2 * j] * r + gpair[2 * j + 1];
            }
        }

        /* pass 3: apply carry + in-block rescan */
        #pragma omp target teams distribute parallel for
        for (int64_t k = 0; k < nb; k++) {
            const int64_t s = k * BLK;
            int64_t e = s + BLK;
            if (e > n) e = n;
            double acc = pair[2 * k] * rc[k / G2] + pair[2 * k + 1];
            for (int64_t i = s; i < e; i++) {
                acc = c[i] * acc + x[i];
                y[i] = acc;
            }
        }
    }
}
