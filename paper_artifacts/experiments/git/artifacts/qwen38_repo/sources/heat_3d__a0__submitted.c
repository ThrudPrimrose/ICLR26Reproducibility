// heat_3d_fp64 -- 3D explicit heat equation, alternating A->B->A 7-point stencil updates.
//
// Optimizations vs. the naive reference:
//   1. The two interior-copy sweeps (Ac / Bc temporaries, each a full N^3 memcpy
//      per timestep) are eliminated: the stencil reads A/B directly with the same
//      offsets, so each timestep is exactly two stencil sweeps instead of four.
//   2. The innermost (k) loop is auto-vectorized (AVX-512 on the target) with FMA
//      contraction; per-point arithmetic keeps the reference's exact op ordering.
//   3. OpenMP: the spatial sweep is parallelized over the (i,j) grid rows with the
//      timesteps kept serial (a true recurrence). Because the per-timestep barrier
//      cost is a fixed overhead, tiny grids are run serially -- a work-based switch
//      picks whichever is faster on the current core count.

#include <stdint.h>
#include <omp.h>

static inline void stencil_row(const double *restrict a, double *restrict b,
                               int64_t Ni, int64_t N, int64_t NN, double alpha) {
    for (int64_t k = 1; k <= Ni; ++k) {
        double c = a[k];
        double tx = alpha * (a[k + NN] - 2.0 * c + a[k - NN]);
        double ty = alpha * (a[k + N] - 2.0 * c + a[k - N]);
        double tz = alpha * (a[k + 1] - 2.0 * c + a[k - 1]);
        b[k] = ((tx + ty) + tz) + c;
    }
}

static void run_serial(double *restrict A, double *restrict B,
                       int64_t N, int64_t TSTEPS, double alpha) {
    const int64_t Ni = N - 2;
    const int64_t NN = N * N;
    for (int64_t t = 1; t <= TSTEPS; ++t) {
        for (int64_t i = 1; i <= Ni; ++i)
            for (int64_t j = 1; j <= Ni; ++j)
                stencil_row(A + i * NN + j * N, B + i * NN + j * N, Ni, N, NN, alpha);
        for (int64_t i = 1; i <= Ni; ++i)
            for (int64_t j = 1; j <= Ni; ++j)
                stencil_row(B + i * NN + j * N, A + i * NN + j * N, Ni, N, NN, alpha);
    }
}

static void run_parallel(double *restrict A, double *restrict B,
                         int64_t N, int64_t TSTEPS, double alpha) {
    const int64_t Ni = N - 2;
    const int64_t NN = N * N;
#pragma omp parallel
    {
        for (int64_t t = 1; t <= TSTEPS; ++t) {
            // sweep 1: B[interior] = f(A)  (reads A, writes B)
#pragma omp for schedule(static)
            for (int64_t i = 1; i <= Ni; ++i)
                for (int64_t j = 1; j <= Ni; ++j)
                    stencil_row(A + i * NN + j * N, B + i * NN + j * N, Ni, N, NN, alpha);
            // sweep 2: A[interior] = f(B)  (reads B, writes A)
#pragma omp for schedule(static)
            for (int64_t i = 1; i <= Ni; ++i)
                for (int64_t j = 1; j <= Ni; ++j)
                    stencil_row(B + i * NN + j * N, A + i * NN + j * N, Ni, N, NN, alpha);
        }
    }
}

void heat_3d_fp64(double *restrict A, double *restrict B,
                  const int64_t N, const int64_t TSTEPS, const double alpha) {
    const int64_t Ni = N - 2;
    if (Ni <= 0 || TSTEPS <= 0) return;

    // Parallelize only when the total number of point-updates is large enough to
    // amortize the 2*TSTEPS per-timestep barriers (and we actually have >1 thread).
    const int64_t total = 2 * TSTEPS * Ni * Ni * Ni;
    if (omp_get_max_threads() > 1 && total >= 40 * 1000 * 1000)
        run_parallel(A, B, N, TSTEPS, alpha);
    else
        run_serial(A, B, N, TSTEPS, alpha);
}
