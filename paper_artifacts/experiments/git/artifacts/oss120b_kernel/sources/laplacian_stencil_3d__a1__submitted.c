/*
 * laplacian_stencil_3d kernel implementation (double precision).
 * Applies an 8th-order central finite-difference Laplacian stencil on a
 * periodic 3D grid for a batch of k wavefunctions and computes the per-state
 * kinetic energy ekin = -0.5 * sum(psi * lap).
 */

#include <stdint.h>
#include <stddef.h>
#include <omp.h>

/* Coefficients for the 8th-order second-derivative stencil (R = 4) */
static const double C0 = -205.0/72.0;                /* central coefficient */
static const double CW[4] = {8.0/5.0, -1.0/5.0, 8.0/315.0, -1.0/560.0};

/*
 * Expected signature for the benchmark harness:
 *   lap   : output Laplacian values (size N*N*N*k)
 *   psi   : input wavefunctions (size N*N*N*k)
 *   ekin  : output kinetic energies per wavefunction (size k)
 *   N     : grid size per spatial dimension
 *   k     : number of wavefunctions (batch size)
 *   inv_h2: inverse squared grid spacing (scalar)
 */
void laplacian_stencil_3d_fp64(double *restrict lap,
                               const double *restrict psi,
                               double *restrict ekin,
                               int64_t N,
                               int64_t k,
                               double inv_h2)
{
    const double center_factor = 3.0 * C0;

    /* First pass: compute Laplacian for every grid point and wavefunction */
    #pragma omp parallel for collapse(3) schedule(static)
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            for (int64_t l = 0; l < N; ++l) {
                int64_t base = ((i * N + j) * N + l) * k;

                /* Pre‑compute neighbour base offsets for each axis and distance */
                int64_t i_off[4];
                int64_t im_off[4];
                int64_t j_off[4];
                int64_t jm_off[4];
                int64_t l_off[4];
                int64_t lm_off[4];
                for (int m = 1; m <= 4; ++m) {
                    int64_t ip = (i + m) % N;
                    int64_t im = (i - m + N) % N;
                    i_off[m-1] = ((ip * N + j) * N + l) * k;
                    im_off[m-1] = ((im * N + j) * N + l) * k;

                    int64_t jp = (j + m) % N;
                    int64_t jm = (j - m + N) % N;
                    j_off[m-1] = ((i * N + jp) * N + l) * k;
                    jm_off[m-1] = ((i * N + jm) * N + l) * k;

                    int64_t lp = (l + m) % N;
                    int64_t lm = (l - m + N) % N;
                    l_off[m-1] = ((i * N + j) * N + lp) * k;
                    lm_off[m-1] = ((i * N + j) * N + lm) * k;
                }

                for (int64_t w = 0; w < k; ++w) {
                    double acc = center_factor * psi[base + w];
                    for (int m = 0; m < 4; ++m) {
                        double wcoeff = CW[m];
                        acc += wcoeff * (psi[i_off[m] + w] + psi[im_off[m] + w]);
                        acc += wcoeff * (psi[j_off[m] + w] + psi[jm_off[m] + w]);
                        acc += wcoeff * (psi[l_off[m] + w] + psi[lm_off[m] + w]);
                    }
                    lap[base + w] = inv_h2 * acc;
                }
            }
        }
    }

    /* Second pass: compute kinetic energy per wavefunction */
    #pragma omp parallel for schedule(static)
    for (int64_t w = 0; w < k; ++w) {
        double sum = 0.0;
        for (int64_t i = 0; i < N; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                for (int64_t l = 0; l < N; ++l) {
                    int64_t idx = ((i * N + j) * N + l) * k + w;
                    sum += psi[idx] * lap[idx];
                }
            }
        }
        ekin[w] = -0.5 * sum;
    }
}

/* End of laplacian_stencil_3d_fp64 */
