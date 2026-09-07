/* WarpX Boris momentum pusher -- optimized C.
 *
 * ABI (canonical HPCAgent-Bench C-ABI): array pointers first (alphabetical),
 * then scalars + size symbols (alphabetical), then reserved workspace tail.
 * dt is the pinned physical constant 1e-13 (not an argument).
 */
#include <stdint.h>
#include <math.h>
#include <omp.h>

/* Keep arithmetic bit-identical to the NumPy oracle: no FMA contraction. */
#pragma GCC optimize ("fp-contract=off")

#define WARPX_INV_C2 1.1126500560536185e-17 /* 1.0/(299792458.0*299792458.0) in fp64 */
#define WARPX_DT     1.0e-13

void warpx_boris_push_fp64(const double *restrict Bx, const double *restrict By,
                      const double *restrict Bz, const double *restrict Ex,
                      const double *restrict Ey, const double *restrict Ez,
                      double *restrict ux, double *restrict uy,
                      double *restrict uz, const double m,
                      const int64_t momentum_push_type,
                      const int64_t np_particles, const double q,
                      uint8_t *workspace, int64_t workspace_size)
{
  (void)workspace; (void)workspace_size;
  const int64_t mpt = (int64_t)momentum_push_type;
  const double econst = ((0.5 * q) * WARPX_DT) / m;
  const double inv_c2 = WARPX_INV_C2;

  #pragma omp parallel for schedule(static)
  for (int64_t ip = 0; ip < np_particles; ++ip) {
    double ux_i = ux[ip], uy_i = uy[ip], uz_i = uz[ip];

    if (mpt == 1 || mpt == 0) {
      ux_i += econst * Ex[ip];
      uy_i += econst * Ey[ip];
      uz_i += econst * Ez[ip];
    }

    const double g = 1.0 / sqrt(1.0 + ((ux_i * ux_i + uy_i * uy_i) + uz_i * uz_i) * inv_c2);
    double tx = (econst * g) * Bx[ip];
    double ty = (econst * g) * By[ip];
    double tz = (econst * g) * Bz[ip];

    if (mpt == 1 || mpt == 2) {
      const double tsq = ((tx * tx + ty * ty) + tz * tz);
      const double factor = (tsq > 0.0) ? ((sqrt(1.0 + tsq) - 1.0) / tsq) : 0.5;
      tx *= factor;
      ty *= factor;
      tz *= factor;
    }

    const double tsqi = 2.0 / (((1.0 + tx * tx) + ty * ty) + tz * tz);
    const double sx = tx * tsqi, sy = ty * tsqi, sz = tz * tsqi;
    const double ux_p = ux_i + uy_i * tz - uz_i * ty;
    const double uy_p = uy_i + uz_i * tx - ux_i * tz;
    const double uz_p = uz_i + ux_i * ty - uy_i * tx;
    ux_i += uy_p * sz - uz_p * sy;
    uy_i += uz_p * sx - ux_p * sz;
    uz_i += ux_p * sy - uy_p * sx;

    if (mpt == 2 || mpt == 0) {
      ux_i += econst * Ex[ip];
      uy_i += econst * Ey[ip];
      uz_i += econst * Ez[ip];
    }

    ux[ip] = ux_i;
    uy[ip] = uy_i;
    uz[ip] = uz_i;
  }
}
