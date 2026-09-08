#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

void scan_affine_decay_fp64(double * __restrict__ y,
                            double * __restrict__ c,
                            double * __restrict__ x,
                            int64_t LEN_1D,
                            uint8_t *workspace,
                            int64_t workspace_bytes)
{
    int64_t n = LEN_1D;
    if (n <= 0) return;

    printf("DEBUG n=%ld y0=%.17g c0=%.17g x0=%.17g\n", n, y[0], c[0], x[0]);
    fflush(stdout);

    y[0] = x[0];
    if (n == 1) return;

    /* Tiny inputs: the serial recurrence is fastest. */
    if (n < 4096) {
        for (int64_t i = 1; i < n; ++i) {
            y[i] = c[i] * y[i - 1] + x[i];
        }
        return;
    }

    /* Target roughly 8 chunks per thread so the scan across blocks stays short
     * while each thread still works on a wide contiguous span. */
    int max_threads = omp_get_max_threads();
    if (max_threads < 1) max_threads = 1;
    int64_t block_size = (n + max_threads * 8 - 1) / (max_threads * 8);
    if (block_size < 1024) block_size = 1024;
    if (block_size > 8192) block_size = 8192;
    int64_t num_blocks = (n + block_size - 1) / block_size;

    #define LOCAL_BLK_MAX 1024
    double local_work[2 * LOCAL_BLK_MAX];
    double *work = local_work;
    int allocated = 0;
    if (num_blocks > LOCAL_BLK_MAX) {
        int64_t need = 2 * num_blocks * sizeof(double);
        if (workspace != NULL && workspace_bytes >= need) {
            work = (double *)(void *)workspace;
        } else {
            work = (double *)malloc(need);
            if (!work) {
                for (int64_t i = 1; i < n; ++i) {
                    y[i] = c[i] * y[i - 1] + x[i];
                }
                return;
            }
            allocated = 1;
        }
    }
    double * __restrict__ blkA = work;
    double * __restrict__ blkB = work + num_blocks;

    #pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int64_t chunk = (num_blocks + nt - 1) / nt;
        int64_t b0 = tid * chunk;
        int64_t b1 = b0 + chunk;
        if (b1 > num_blocks) b1 = num_blocks;

        /* Pass 1: each block computes its local scan assuming a zero seed. */
        for (int64_t b = b0; b < b1; ++b) {
            int64_t lo = b * block_size;
            int64_t hi = lo + block_size;
            if (hi > n) hi = n;

            double prod = 1.0;
            double z = 0.0;
            for (int64_t i = lo; i < hi; ++i) {
                double ci = c[i];
                prod *= ci;
                z = ci * z + x[i];
                y[i] = z;
            }
            blkA[b] = prod;
            blkB[b] = z;          /* tail value for a zero seed */
        }

        #pragma omp barrier

        /* Pass 2: serial affine scan across the block tails (one thread). */
        #pragma omp single
        {
            double carry = 0.0;
            for (int64_t b = 0; b < num_blocks; ++b) {
                double new_carry = blkA[b] * carry + blkB[b];
                blkB[b] = carry;
                carry = new_carry;
            }
        }

        /* Pass 3: add the correct prefix contribution to every element. */
        for (int64_t b = b0; b < b1; ++b) {
            int64_t lo = b * block_size;
            int64_t hi = lo + block_size;
            if (hi > n) hi = n;

            double prod = 1.0;
            double start = blkB[b];
            for (int64_t i = lo; i < hi; ++i) {
                prod *= c[i];
                y[i] += prod * start;
            }
        }
    }

    if (allocated) {
        free(work);
    }
}
