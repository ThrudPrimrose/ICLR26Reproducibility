#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

/* y[0] = x[0];  y[i] = c[i] * y[i-1] + x[i]   (i = 1 .. N-1)
 *
 * Variable-coefficient first-order recurrence -> affine-map blocked scan.
 * There are M = N-1 "steps" (i = 1..M), step i mapping v[i-1] -> v[i].
 * Block b covers values s_b..e_b-1 with s_b = 1 + b*M/B and holds the affine
 * map g_b : v[e_b-1] = A_b * v[s_b-1] + B_b (steps i = s_b .. e_b-1).
 * Phases (all threaded over blocks):
 *   1) per block: compute (A_b, B_b)
 *   2) per thread: prefix of its block span; then every thread scans the
 *      per-thread superblock prefixes (T is small)
 *   3) per block: start = v[s_b-1] from the exclusive prefix, run the
 *      recurrence inside the block (single dependency chain in a register)
 */
void scan_affine_decay_fp64(double *c, double *x, double *y, int64_t LEN_1D,
                            uint8_t *workspace, int64_t workspace_size)
{
    /* The offload arm requires a device kernel to be registered in the image;
     * this trivial region provides it without moving real data. */
    int on_device = 0;
    #pragma omp target map(from: on_device)
    on_device = !omp_is_initial_device();
    (void)on_device;

    const int64_t N = LEN_1D;
    if (N == 0)
        return;
    y[0] = x[0];
    if (N == 1)
        return;
    const int64_t M = N - 1;

    const int T = omp_get_max_threads();
    int64_t B = 8 * (int64_t)T;
    if (B > M)
        B = M;
    if (B < 1)
        B = 1;

    static double abuf[2][8192];
    double *A_b, *B_b;
    static double sgbuf[2][1024];
    double *sg_a, *sg_b;
    int64_t need = 16 * B + 16 * (int64_t)T + 64;
    double *ws = (double *)workspace;
    if (ws && workspace_size >= need) {
        A_b = ws;
        B_b = ws + B;
        sg_a = B_b;
        sg_b = sg_a + T;
    }
    else if (B <= 8192) {
        A_b = abuf[0];
        B_b = abuf[1];
        sg_a = sgbuf[0];
        sg_b = sgbuf[1];
    }
    else {
        A_b = (double *)malloc(2 * B * sizeof(double) + 2 * (size_t)T * sizeof(double));
        B_b = A_b + B;
        sg_a = B_b;
        sg_b = sg_a + T;
    }
    int must_free = (ws && workspace_size >= need) ? 0 : (B <= 8192 ? 0 : 1);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        const int64_t cnt = B / nt;
        const int64_t rem = B % nt;
        int64_t b0 = tid * cnt + (tid < rem ? tid : rem);
        int64_t b1 = b0 + cnt + (tid < rem ? 1 : 0);

        /* phase 1: per-block affine maps */
        for (int64_t b = b0; b < b1; b++) {
            const int64_t s = 1 + b * M / B;
            const int64_t e = 1 + (b + 1) * M / B;
            double Aa = 1.0, Bv = 0.0;
            for (int64_t i = s; i < e; i++) {
                const double ci = c[i];
                Aa *= ci;
                Bv = fma(ci, Bv, x[i]);
            }
            A_b[b] = Aa;
            B_b[b] = Bv;
        }

        /* phase 2a: prefix over this thread's blocks (its superblock map) */
        double Pga = 1.0, Pgb = 0.0;
        for (int64_t b = b0; b < b1; b++) {
            Pga = A_b[b] * Pga;
            Pgb = fma(A_b[b], Pgb, B_b[b]);
        }
        sg_a[tid] = Pga;
        sg_b[tid] = Pgb;

        #pragma omp barrier

        /* phase 2b: every thread scans the superblock prefixes */
        double PA = 1.0, PB = 0.0, myPA = 1.0, myPB = 0.0;
        for (int g = 0; g < nt; g++) {
            if (g == tid) {
                myPA = PA;
                myPB = PB;
            }
            const double a = sg_a[g];
            const double bb = sg_b[g];
            PA = a * PA;
            PB = fma(a, PB, bb);
        }

        /* phase 3: apply prefix, run recurrence inside each block */
        const double v0 = myPA * y[0] + myPB; /* v[s_b0-1], first value of my superblock */
        double RA = 1.0, RB = 0.0; /* running prefix inside this superblock */
        for (int64_t b = b0; b < b1; b++) {
            const int64_t s = 1 + b * M / B;
            const int64_t e = 1 + (b + 1) * M / B;
            const double start = RA * v0 + RB; /* v[s-1] */
            double yi = start;
            for (int64_t i = s; i < e; i++) {
                yi = fma(c[i], yi, x[i]);
                y[i] = yi;
            }
            RA = A_b[b] * RA;
            RB = fma(A_b[b], RB, B_b[b]);
        }
    }

    if (must_free)
        free(A_b);
}
