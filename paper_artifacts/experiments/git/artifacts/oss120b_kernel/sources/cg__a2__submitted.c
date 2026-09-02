// Conjugate Gradient (CG) solver for symmetric positive-definite matrix in CSR format.
// Implements the reference algorithm with OpenMP parallelism for performance.
// Function name follows the benchmark convention: <kernel>_fp64.
// Arguments:
//   A_indptr: row pointer array (size N+1) of CSR matrix (int64).
//   A_indices: column indices array (size nnz) of CSR matrix (int64).
//   A_data: non-zero values array (size nnz) of CSR matrix (double).
//   x: solution vector, also input initial guess (size N) (in/out).
//   b: right-hand side vector (size N) (in).
//   N: dimension of the system (rows of A).
//   max_iter: maximum number of CG iterations.
//   nnz: number of non-zero entries (unused).
//   tol: tolerance for convergence (stops when sqrt(rsnew) < tol).

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

// Function signature as expected by the harness.
void cg_csr_fp64(const double *restrict A_data,
                 const int64_t *restrict A_indices,
                 const int64_t *restrict A_indptr,
                 const double *restrict b,
                 double *restrict x,
                 const int64_t N,
                 const int64_t max_iter,
                 const int64_t nnz,
                 const double tol) {
    // Allocate temporary vectors.
    //
    double *r = (double *)malloc((size_t)N * sizeof(double));
    double *p = (double *)malloc((size_t)N * sizeof(double));
    double *Ap = (double *)malloc((size_t)N * sizeof(double));
    if (!r || !p || !Ap) {
        // Allocation failure - abort.
        exit(1);
    }

    // Compute initial residual r = b - A * x, and set initial search direction p = r
    //
    for (int64_t i = 0; i < N; ++i) {
        double sum = 0.0;
                for (int64_t k = A_indptr[i]; k < A_indptr[i + 1]; ++k) {
            sum += A_data[k] * x[A_indices[k]];
        }
        double ri = b[i] - sum;
        r[i] = ri;
        p[i] = ri;
    }

    // Compute initial dot product rsold = r' * r
    double rsold = 0.0;
    #pragma omp parallel for reduction(+:rsold) schedule(static)
    for (int64_t i = 0; i < N; ++i) {
        rsold += r[i] * r[i];
    }

    for (int64_t iter = 0; iter < max_iter; ++iter) {
        // Compute Ap = A * p and pAp = p' * Ap in one pass
        double pAp = 0.0;
        // reduction(+:pAp)
        for (int64_t i = 0; i < N; ++i) {
            double sum = 0.0;
                        for (int64_t k = A_indptr[i]; k < A_indptr[i + 1]; ++k) {
                sum += A_data[k] * p[A_indices[k]];
            }
            Ap[i] = sum;
            pAp += p[i] * sum;
        }

        double alpha = rsold / pAp;

        // Update x, r, and compute new rsnew = r' * r in one pass
        double rsnew = 0.0;
        // reduction(+:rsnew)
        for (int64_t i = 0; i < N; ++i) {
            x[i] += alpha * p[i];
            double ri = r[i] - alpha * Ap[i];
            r[i] = ri;
            rsnew += ri * ri;
        }

        if (sqrt(rsnew) < tol) {
            break;
        }

        double beta = rsnew / rsold;

        // Update search direction p = r + beta * p
        //
        for (int64_t i = 0; i < N; ++i) {
            p[i] = r[i] + beta * p[i];
        }

        rsold = rsnew;
    }

    free(r);
    free(p);
    free(Ap);
}
