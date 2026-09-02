// Conjugate Gradient (CG) solve for A x = b, A in CSR (fp64) -- optimized in place.
//
// C-ABI per signature.json: symbol cg_csr_fp64, args (A_data, A_indices, A_indptr, b, x, N,
// nnz) plus the trailing Sec.11 workspace pair (uint8_t* workspace, int64_t workspace_size).
//
// The Krylov sweep is a genuine recurrence (each iterate depends on the previous), so the
// iteration loop stays serial; the parallel work is the per-iteration SpMV, the dot products
// and the axpy updates. The dominant cost is the SpMV (streaming A_data + A_indices), so we
// (1) parallelize it over rows with OpenMP and (2) convert the int64 indices to int32 once,
// cutting the per-SpMV index traffic 8->4 bytes (N < 2^31, so exact).
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define CG_MAX_ITER 100
#define CG_TOL 1e-6

static inline double *al64(uint8_t *p) { return (double *)(((uintptr_t)p + 63) & ~(uintptr_t)63); }

// y = A * v (CSR). Parallel over rows; each row's accumulation keeps CSR order, so the
// per-row result matches the reference bit-for-bit (up to the FMA/rounding the reference also
// has). A_idx is the int32 copy of the indices.
static void cg_spmv(const double *__restrict A_data,
                    const int32_t *__restrict A_idx,
                    const int64_t *__restrict A_indptr,
                    const double *__restrict v,
                    double *__restrict y, int64_t N) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; ++i) {
        double acc = 0.0;
        const int64_t k0 = A_indptr[i], k1 = A_indptr[i + 1];
        for (int64_t k = k0; k < k1; ++k)
            acc += A_data[k] * v[A_idx[k]];
        y[i] = acc;
    }
}

static double cg_dot(const double *__restrict a, const double *__restrict b, int64_t N) {
    double s = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:s)
    for (int64_t i = 0; i < N; ++i) s += a[i] * b[i];
    return s;
}

void cg_csr_fp64(const double *__restrict A_data,
                 const int64_t *__restrict A_indices,
                 const int64_t *__restrict A_indptr,
                 const double *__restrict b,
                 double *__restrict x,
                 int64_t N, int64_t nnz,
                 uint8_t *__restrict workspace, int64_t workspace_size) {
    // scratch: r, p, Ap (3*N doubles) + idx32 (nnz int32), each 64B-aligned.
    const int64_t need = 24 * N + 4 * nnz + 256;
    double *r, *p, *Ap;
    int32_t *idx32;
    const int use_ws = (workspace != NULL) && (workspace_size >= need);
    if (use_ws) {
        r   = al64(workspace);
        p   = al64((uint8_t*)r  + 8 * N);
        Ap  = al64((uint8_t*)p  + 8 * N);
        idx32 = (int32_t*)(((uintptr_t)Ap + 8 * N + 63) & ~(uintptr_t)63);
    } else {
        const size_t db = (size_t)N * sizeof(double);
        r  = (double*)aligned_alloc(64, db);
        p  = (double*)aligned_alloc(64, db);
        Ap = (double*)aligned_alloc(64, db);
        idx32 = (int32_t*)aligned_alloc(64, (size_t)nnz * sizeof(int32_t));
    }

    // int64 -> int32 index copy (exact: N < 2^31).
    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < nnz; ++k) idx32[k] = (int32_t)A_indices[k];

    // r = b - A*x ; p = r ; rsold = r.r
    cg_spmv(A_data, idx32, A_indptr, x, Ap, N);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; ++i) r[i] = b[i] - Ap[i];
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; ++i) p[i] = r[i];
    double rsold = cg_dot(r, r, N);

    for (int it = 0; it < CG_MAX_ITER; ++it) {
        cg_spmv(A_data, idx32, A_indptr, p, Ap, N);
        double pAp = cg_dot(p, Ap, N);
        double alpha = rsold / pAp;
        double rsnew = 0.0;
        #pragma omp parallel for schedule(static) reduction(+:rsnew)
        for (int64_t i = 0; i < N; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
            rsnew += r[i] * r[i];
        }
        if (sqrt(rsnew) < CG_TOL) break;
        double beta = rsnew / rsold;
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < N; ++i) p[i] = r[i] + beta * p[i];
        rsold = rsnew;
    }

    if (!use_ws) { free(r); free(p); free(Ap); free(idx32); }
}
