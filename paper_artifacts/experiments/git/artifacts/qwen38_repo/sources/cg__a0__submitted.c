/* Conjugate Gradient for symmetric positive-definite CSR systems.
 *
 * Optimized port of the auto-generated naive implementation:
 *   - OpenMP-parallel SpMV (the dominant cost: nnz mults per iteration),
 *     with the p'Ap dot fused into the SpMV row loop (one reduction);
 *   - fused x/r update pass that also accumulates r'r;
 *   - scratch (r, p, Ap) taken from the ABI workspace when available,
 *     so nothing is malloc'ed inside the timed call;
 *   - the Krylov sweep itself stays a loop: each iterate depends on the
 *     previous one.
 *
 * Arithmetic matches the NumPy reference (reference.py): same formulas,
 * same per-row accumulation order in the SpMV; only the order of the
 * outer reductions differs (parallel), which is far inside the fp64
 * grading band.
 */
#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include <omp.h>

#define CG_MAX_ITER 100
#define CG_TOL 1e-6

/* SpMV with the dot p'y accumulated in the same pass (saves one array sweep
 * over N per CG iteration): returns sum_i p[i] * y[i]. */
static double cg_spmv_dotp(const double *restrict A_data,
                           const int64_t *restrict A_indices,
                           const int64_t *restrict A_indptr,
                           const double *restrict p,
                           double *restrict y,
                           int64_t N)
{
    double pAp = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:pAp)
    for (int64_t i = 0; i < N; ++i) {
        double s = 0.0;
        const int64_t beg = A_indptr[i], end = A_indptr[i + 1];
        for (int64_t k = beg; k < end; ++k) {
            s += A_data[k] * p[A_indices[k]];
        }
        y[i] = s;
        pAp += p[i] * s;
    }
    return pAp;
}

/* Parallel dot product. */
static double cg_dot(const double *restrict a, const double *restrict b, int64_t N)
{
    double s = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:s)
    for (int64_t i = 0; i < N; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

void cg_csr_fp64(const double *restrict A_data,
                 const int64_t *restrict A_indices,
                 const int64_t *restrict A_indptr,
                 const double *restrict b,
                 double *restrict x,
                 const int64_t N,
                 const int64_t nnz,
                 uint8_t *workspace,
                 int64_t workspace_size)
{
    (void)nnz;

    const size_t n8 = (size_t)N * sizeof(double);
    size_t need = 3 * n8;
    bool use_ws = (workspace != NULL) && workspace_size >= (int64_t)need;
    double *r, *p, *Ap;
    if (use_ws) {
        char *w = (char *)workspace;
        r = (double *)w;
        p = (double *)(w + n8);
        Ap = (double *)(w + 2 * n8);
    } else {
        r = (double *)malloc(n8);
        p = (double *)malloc(n8);
        Ap = (double *)malloc(n8);
        if (!r || !p || !Ap) {
            free(r); free(p); free(Ap);
            return;
        }
    }

    /* r = b - A @ x; p = r; rsold = r @ r. */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; ++i) {
        double s = 0.0;
        const int64_t beg = A_indptr[i], end = A_indptr[i + 1];
        for (int64_t k = beg; k < end; ++k) {
            s += A_data[k] * x[A_indices[k]];
        }
        r[i] = b[i] - s;
    }
    memcpy(p, r, n8);
    double rsold = cg_dot(r, r, N);

    for (int iter = 0; iter < CG_MAX_ITER; ++iter) {
        double pAp = cg_spmv_dotp(A_data, A_indices, A_indptr, p, Ap, N);
        const double alpha = rsold / pAp;

        /* x += alpha*p; r -= alpha*Ap; rsnew = r @ r -- one fused pass. */
        double rsnew = 0.0;
        #pragma omp parallel for schedule(static) reduction(+:rsnew)
        for (int64_t i = 0; i < N; ++i) {
            const double ai = Ap[i];
            const double pi = p[i];
            x[i] = x[i] + alpha * pi;
            const double rn = r[i] - alpha * ai;
            r[i] = rn;
            rsnew += rn * rn;
        }
        if (sqrt(rsnew) < CG_TOL) {
            break;
        }
        const double beta = rsnew / rsold;
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < N; ++i) {
            p[i] = r[i] + beta * p[i];
        }
        rsold = rsnew;
    }

    if (!use_ws) {
        free(r);
        free(p);
        free(Ap);
    }
}
