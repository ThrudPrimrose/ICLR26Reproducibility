/*
 * C implementation of the 3-D Laplacian stencil kernel.
 *
 * The reference Python implementation (laplacian_stencil_3d_numpy.py) computes
 * a high-order finite-difference Laplacian on a periodic N^3 grid for a batch of
 * k wavefunctions. It also evaluates the per-state kinetic energy
 *   ekin[j] = -0.5 * sum_{i,j,l} psi(i,j,l,k) * lap(i,j,l,k).
 *
 * This file provides a C version matching the reference kernel signature and
 * behavior. The interface follows the conventions used by other kernels in the
 * benchmark suite (e.g., jacobi_2d_fp64, heat_3d_fp64):
 *
 *   void laplacian_stencil_3d_fp64(double inv_h2,
 *                                 const double *restrict psi,
 *                                 double *restrict lap,
 *                                 double *restrict ekin,
 *                                 int64_t N,
 *                                 int64_t k);
 *
 * Arguments:
 *   inv_h2  - 1/h^2, grid spacing factor (scalar).
 *   psi    - Input wavefunctions, shape (N, N, N, k) stored in C order.
 *   lap    - Output Laplacian, same shape as psi.
 *   ekin   - Output kinetic-energy vector, length k.
 *   N      - Grid size along each spatial dimension.
 *   k      - Number of wavefunctions (batch size).
 *
 * The computation mirrors the NumPy version exactly, using periodic (wrap-around)
 * indexing to implement np.roll. All arithmetic is performed in double
 * precision. The implementation uses OpenMP to parallelise over the outermost
 * spatial dimension. Per-state kinetic energy accumulation is performed with an
 * OpenMP array reduction to avoid race conditions.
 */

#include <stdint.h>
#include <stddef.h>
#include <omp.h>

static const double C0 = -205.0 / 72.0;
static const double CW[4] = {8.0/5.0, -1.0/5.0, 8.0/315.0, -1.0/560.0};
static const double factor = 3.0 * C0;

void laplacian_stencil_3d_fp64(double inv_h2, const double *restrict psi, double *restrict lap, double *restrict ekin, int64_t N, int64_t k) {
    // Initialise ekin to zero
    for (int64_t s = 0; s < k; ++s) {
        ekin[s] = 0.0;
    }
    size_t stride_i = (size_t)N * N * k;
    size_t stride_j = (size_t)N * k;
    size_t stride_l = (size_t)k;
    #pragma omp parallel for schedule(static) reduction(+: ekin[:k])
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            for (int64_t l = 0; l < N; ++l) {
                size_t base = (size_t)i * stride_i + (size_t)j * stride_j + (size_t)l * stride_l;
                for (int64_t s = 0; s < k; ++s) {
                    double acc = factor * psi[base + s];
                    // axis 0 (i)
                    for (int m = 0; m < 4; ++m) {
                        double w = CW[m];
                        int64_t shift = m + 1;
                        int64_t ip = (i + shift) % N;
                        int64_t im = ((i - shift) % N + N) % N;
                        size_t idx_ip = (size_t)ip * stride_i + (size_t)j * stride_j + (size_t)l * stride_l + s;
                        size_t idx_im = (size_t)im * stride_i + (size_t)j * stride_j + (size_t)l * stride_l + s;
                        acc += w * (psi[idx_ip] + psi[idx_im]);
                    }
                    // axis 1 (j)
                    for (int m = 0; m < 4; ++m) {
                        double w = CW[m];
                        int64_t shift = m + 1;
                        int64_t jp = (j + shift) % N;
                        int64_t jm = ((j - shift) % N + N) % N;
                        size_t idx_jp = (size_t)i * stride_i + (size_t)jp * stride_j + (size_t)l * stride_l + s;
                        size_t idx_jm = (size_t)i * stride_i + (size_t)jm * stride_j + (size_t)l * stride_l + s;
                        acc += w * (psi[idx_jp] + psi[idx_jm]);
                    }
                    // axis 2 (l)
                    for (int m = 0; m < 4; ++m) {
                        double w = CW[m];
                        int64_t shift = m + 1;
                        int64_t lp = (l + shift) % N;
                        int64_t lm = ((l - shift) % N + N) % N;
                        size_t idx_lp = (size_t)i * stride_i + (size_t)j * stride_j + (size_t)lp * stride_l + s;
                        size_t idx_lm = (size_t)i * stride_i + (size_t)j * stride_j + (size_t)lm * stride_l + s;
                        acc += w * (psi[idx_lp] + psi[idx_lm]);
                    }
                    double lap_val = inv_h2 * acc;
                    lap[base + s] = lap_val;
                    /* ekin accumulation moved to a second pass */
                }
            }
        }
    }
    // Compute ekin as -0.5 * sum psi * lap over all grid points
    for (int64_t s = 0; s < k; ++s) {
        ekin[s] = 0.0;
    }
    #pragma omp parallel for schedule(static) reduction(+: ekin[:k])
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            for (int64_t l = 0; l < N; ++l) {
                size_t base = (size_t)i * stride_i + (size_t)j * stride_j + (size_t)l * stride_l;
                for (int64_t s = 0; s < k; ++s) {
                    ekin[s] += psi[base + s] * lap[base + s];
                }
            }
        }
    }
    for (int64_t s = 0; s < k; ++s) {
        ekin[s] = -0.5 * ekin[s];
    }
}
