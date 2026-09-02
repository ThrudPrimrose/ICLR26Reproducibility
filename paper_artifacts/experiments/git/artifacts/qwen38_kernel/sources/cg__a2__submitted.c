#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* pinned config values (compile-time constants per the manifest) */
#define CG_MAX_ITER 100
#define CG_TOL 1.0e-6

static double *g_scratch = NULL;   /* persistent fallback scratch (untimed warmup covers first touch) */
static size_t g_scratch_bytes = 0;

static double *scratch_get(size_t bytes)
{
    if (bytes > g_scratch_bytes) {
        double *p = (double *)realloc(g_scratch, bytes);
        if (!p) return NULL;
        g_scratch = p;
        g_scratch_bytes = bytes;
    }
    return g_scratch;
}

static void spmv(const double *restrict A_data,
                 const int64_t *restrict A_indices,
                 const int64_t *restrict A_indptr,
                 const double *restrict x,
                 double *restrict y,
                 const int64_t N)
{
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; i++) {
        double s = 0.0;
        int64_t j = A_indptr[i], e = A_indptr[i + 1];
        for (; j < e; j++)
            s += A_data[j] * x[A_indices[j]];
        y[i] = s;
    }
}

static double dot2(const double *restrict a, const double *restrict c, const int64_t N)
{
    double s = 0.0;
    #pragma omp parallel for reduction(+:s) schedule(static)
    for (int64_t i = 0; i < N; i++)
        s += a[i] * c[i];
    return s;
}

void cg_csr_fp64(const double *restrict A_data,
                 const int64_t *restrict A_indices,
                 const int64_t *restrict A_indptr,
                 const double *restrict b,
                 double *restrict x,
                 const int64_t N,
                 const int64_t nnz,
                 uint8_t *restrict workspace,
                 const int64_t workspace_size)
{
    (void)nnz;
    const int64_t max_iter = CG_MAX_ITER;
    const double tol = CG_TOL;

    size_t need = (size_t)N * 24; /* r, p, Ap : 3 x 8N */
    double *base;
    if (workspace && (size_t)workspace_size >= need) {
        base = (double *)workspace;
    } else {
        base = scratch_get(need);
    }
    double *r = base, *p = r + N, *Ap = p + N;

    spmv(A_data, A_indices, A_indptr, x, Ap, N);          /* Ap = A x0 */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; i++) r[i] = b[i] - Ap[i];   /* r = b - A x0 */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; i++) p[i] = r[i];           /* p = r */
    double rsold = dot2(r, r, N);

    for (int64_t it = 0; it < max_iter; it++) {
        spmv(A_data, A_indices, A_indptr, p, Ap, N);
        double pAp = dot2(p, Ap, N);
        double alpha = rsold / pAp;
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < N; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        double rsnew = dot2(r, r, N);
        if (sqrt(rsnew) < tol) break;
        double beta = rsnew / rsold;
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < N; i++) p[i] = r[i] + beta * p[i];
        rsold = rsnew;
    }
}
