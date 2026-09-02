#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/*
 * Conjugate Gradient solver for symmetric positive-definite sparse matrix A in CSR format.
 * A is stored in three arrays: A_data (non-zero values), A_indices (column indices), A_indptr (row pointers).
 * The vectors x and b are of length N.
 * The solver runs up to max_iter iterations or until the residual norm falls below tol.
 * All arrays are assumed to be contiguous and of appropriate size.
 */

void cg_fp64(const double *restrict A_data,
             const int64_t *restrict A_indices,
             const int64_t *restrict A_indptr,
             double *restrict x,
             const double *restrict b,
             int64_t N,
             int64_t max_iter,
             double tol) {
    // Allocate temporary vectors
    double *r = (double *)malloc((size_t)N * sizeof(double));
    double *p = (double *)malloc((size_t)N * sizeof(double));
    double *Ap = (double *)malloc((size_t)N * sizeof(double));
    if (!r || !p || !Ap) {
        // allocation failed, abort
        free(r);
        free(p);
        free(Ap);
        return;
    }
    double rsold = 0.0;

    // Compute initial residual r = b - A*x
    // Compute Ap = A*x in temporary to reuse code
    // First compute Ax into Ap
    #pragma omp parallel for reduction(+:rsold) schedule(static)
    for (int64_t i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int64_t jj = A_indptr[i]; jj < A_indptr[i+1]; ++jj) {
            sum += A_data[jj] * x[A_indices[jj]];
        }
        Ap[i] = sum;
        double ri = b[i] - sum;
        r[i] = ri;
        p[i] = ri;
        rsold += ri * ri;
    }

    for (int64_t iter = 0; iter < max_iter; ++iter) {
        // Compute Ap = A * p
        double pAp = 0.0;
        #pragma omp parallel for reduction(+:pAp) schedule(static)
        for (int64_t i = 0; i < N; ++i) {
            double sum = 0.0;
            for (int64_t jj = A_indptr[i]; jj < A_indptr[i+1]; ++jj) {
                sum += A_data[jj] * p[A_indices[jj]];
            }
            Ap[i] = sum;
            pAp += p[i] * sum;
        }
        double alpha = rsold / pAp;
        // Update x = x + alpha * p, update r = r - alpha * Ap, and compute rsnew = r^T r in one pass
        double rsnew = 0.0;
        #pragma omp parallel for reduction(+:rsnew) schedule(static)
        for (int64_t i = 0; i < N; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
            rsnew += r[i] * r[i];
        }
        if (sqrt(rsnew) < tol) {
            break;
        }
        double beta = rsnew / rsold;
        // Update p = r + beta * p
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < N; ++i) {
            p[i] = r[i] + beta * p[i];
        }
        rsold = rsnew;
    }

    // Clean up
    free(r);
    free(p);
    free(Ap);
}
