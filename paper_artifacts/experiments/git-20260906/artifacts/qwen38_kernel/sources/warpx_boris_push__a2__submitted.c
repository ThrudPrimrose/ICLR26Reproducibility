/* WarpX Boris particle-momentum pusher -- optimized C.
 *
 * Elementwise over particles; the per-particle math matches the NumPy reference
 * line-for-line (same op order, no fused multiply-add), so results are bit-identical
 * to the oracle at any thread/vector width.
 *
 * C-ABI (c-abi-v2): pointers sorted by name, then scalars sorted by name, then the
 * reserved workspace pair.  dt is pinned in the manifest (1.0e-13) and is not an
 * argument; m, momentum_push_type, np_particles and q are.
 */
#pragma GCC optimize ("-ffp-contract=off")
#include <math.h>

static const double C_LIGHT = 299792458.0;
static const double INV_C2 = 1.0 / (C_LIGHT * C_LIGHT);
static const double DT = 1.0e-13;   /* pinned manifest constant */

void warpx_boris_push_fp64(const double *__restrict__ Bx,
                           const double *__restrict__ By,
                           const double *__restrict__ Bz,
                           const double *__restrict__ Ex,
                           const double *__restrict__ Ey,
                           const double *__restrict__ Ez,
                           double *__restrict__ ux,
                           double *__restrict__ uy,
                           double *__restrict__ uz,
                           double m,
                           long momentum_push_type,
                           long np_particles,
                           double q,
                           unsigned char *__restrict__ workspace,
                           long workspace_size)
{
  (void)workspace; (void)workspace_size;

  const double econst = 0.5 * q * DT / m;
  const int do_first = (momentum_push_type == 1 || momentum_push_type == 0);
  const int do_rescale = (momentum_push_type == 1 || momentum_push_type == 2);
  const int do_second = (momentum_push_type == 2 || momentum_push_type == 0);

  #pragma omp parallel for schedule(static)
  for (long ip = 0; ip < np_particles; ++ip) {
    double uxv = ux[ip], uyv = uy[ip], uzv = uz[ip];

    if (do_first) {
      uxv += econst * Ex[ip];
      uyv += econst * Ey[ip];
      uzv += econst * Ez[ip];
    }

    const double inv_gamma = 1.0 / sqrt(1.0 + (uxv * uxv + uyv * uyv + uzv * uzv) * INV_C2);

    double tx = econst * inv_gamma * Bx[ip];
    double ty = econst * inv_gamma * By[ip];
    double tz = econst * inv_gamma * Bz[ip];

    if (do_rescale) {
      const double tsq = tx * tx + ty * ty + tz * tz;
      const double factor = (tsq > 0.0) ? (sqrt(1.0 + tsq) - 1.0) / tsq : 0.5;
      tx *= factor;
      ty *= factor;
      tz *= factor;
    }

    const double tsqi = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz);
    const double sx = tx * tsqi;
    const double sy = ty * tsqi;
    const double sz = tz * tsqi;
    const double ux_p = uxv + uyv * tz - uzv * ty;
    const double uy_p = uyv + uzv * tx - uxv * tz;
    const double uz_p = uzv + uxv * ty - uyv * tx;
    ux[ip] = uxv + (uy_p * sz - uz_p * sy);
    uy[ip] = uyv + (uz_p * sx - ux_p * sz);
    uz[ip] = uzv + (ux_p * sy - uy_p * sx);

    if (do_second) {
      ux[ip] += econst * Ex[ip];
      uy[ip] += econst * Ey[ip];
      uz[ip] += econst * Ez[ip];
    }
  }
}
