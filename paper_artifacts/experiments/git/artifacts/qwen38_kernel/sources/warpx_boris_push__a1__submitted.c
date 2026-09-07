/* Optimized WarpX Boris particle-momentum pusher (single pass, OpenMP).
 *
 * Faithful port of UpdateMomentumBoris (WarpX, BSD-3-Clause-LBNL): same
 * arithmetic, same order of operations, so results match the NumPy/C++
 * references to within rounding. All three MomentumPushType paths are kept.
 *
 * The per-particle update is an elementwise map with no cross-particle
 * dependence, so the single fused loop is bit-identical for any thread count
 * and any schedule.
 */
#include <math.h>
#include <stdint.h>

#define C_LIGHT 299792458.0
#define DT_FIXED 1.0e-13

void warpx_boris_push_fp64(const double *__restrict__ Bx, const double *__restrict__ By,
                           const double *__restrict__ Bz, const double *__restrict__ Ex,
                           const double *__restrict__ Ey, const double *__restrict__ Ez,
                           double *__restrict__ ux, double *__restrict__ uy,
                           double *__restrict__ uz, const double m,
                           const int64_t momentum_push_type, const int64_t np_particles,
                           const double q) {
  const double econst = (0.5 * q) * DT_FIXED / m;
  const double inv_c2 = 1.0 / (C_LIGHT * C_LIGHT);

  if (momentum_push_type == 0) {
    /* ---- Full push ---- */
    _Pragma("omp parallel for schedule(static)")
    for (int64_t ip = 0; ip < np_particles; ++ip) {
      const double u0 = ux[ip] + econst * Ex[ip];
      const double v0 = uy[ip] + econst * Ey[ip];
      const double w0 = uz[ip] + econst * Ez[ip];
      const double inv_gamma =
          1.0 / sqrt(1.0 + (u0 * u0 + v0 * v0 + w0 * w0) * inv_c2);
      const double tx = econst * inv_gamma * Bx[ip];
      const double ty = econst * inv_gamma * By[ip];
      const double tz = econst * inv_gamma * Bz[ip];
      const double tsqi = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz);
      const double sx = tx * tsqi;
      const double sy = ty * tsqi;
      const double sz = tz * tsqi;
      const double ux_p = u0 + v0 * tz - w0 * ty;
      const double uy_p = v0 + w0 * tx - u0 * tz;
      const double uz_p = w0 + u0 * ty - v0 * tx;
      ux[ip] = u0 + uy_p * sz - uz_p * sy + econst * Ex[ip];
      uy[ip] = v0 + uz_p * sx - ux_p * sz + econst * Ey[ip];
      uz[ip] = w0 + ux_p * sy - uy_p * sx + econst * Ez[ip];
    }
  } else if (momentum_push_type == 1) {
    /* ---- First half push ---- */
    _Pragma("omp parallel for schedule(static)")
    for (int64_t ip = 0; ip < np_particles; ++ip) {
      const double u0 = ux[ip] + econst * Ex[ip];
      const double v0 = uy[ip] + econst * Ey[ip];
      const double w0 = uz[ip] + econst * Ez[ip];
      const double inv_gamma =
          1.0 / sqrt(1.0 + (u0 * u0 + v0 * v0 + w0 * w0) * inv_c2);
      double tx = econst * inv_gamma * Bx[ip];
      double ty = econst * inv_gamma * By[ip];
      double tz = econst * inv_gamma * Bz[ip];
      const double tsq = tx * tx + ty * ty + tz * tz;
      const double factor = (tsq > 0.0) ? (sqrt(1.0 + tsq) - 1.0) / tsq : 0.5;
      tx *= factor;
      ty *= factor;
      tz *= factor;
      const double tsqi = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz);
      const double sx = tx * tsqi;
      const double sy = ty * tsqi;
      const double sz = tz * tsqi;
      const double ux_p = u0 + v0 * tz - w0 * ty;
      const double uy_p = v0 + w0 * tx - u0 * tz;
      const double uz_p = w0 + u0 * ty - v0 * tx;
      ux[ip] = u0 + uy_p * sz - uz_p * sy;
      uy[ip] = v0 + uz_p * sx - ux_p * sz;
      uz[ip] = w0 + ux_p * sy - uy_p * sx;
    }
  } else {
    /* ---- Second half push ---- */
    _Pragma("omp parallel for schedule(static)")
    for (int64_t ip = 0; ip < np_particles; ++ip) {
      const double u0 = ux[ip];
      const double v0 = uy[ip];
      const double w0 = uz[ip];
      const double inv_gamma =
          1.0 / sqrt(1.0 + (u0 * u0 + v0 * v0 + w0 * w0) * inv_c2);
      double tx = econst * inv_gamma * Bx[ip];
      double ty = econst * inv_gamma * By[ip];
      double tz = econst * inv_gamma * Bz[ip];
      const double tsq = tx * tx + ty * ty + tz * tz;
      const double factor = (tsq > 0.0) ? (sqrt(1.0 + tsq) - 1.0) / tsq : 0.5;
      tx *= factor;
      ty *= factor;
      tz *= factor;
      const double tsqi = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz);
      const double sx = tx * tsqi;
      const double sy = ty * tsqi;
      const double sz = tz * tsqi;
      const double ux_p = u0 + v0 * tz - w0 * ty;
      const double uy_p = v0 + w0 * tx - u0 * tz;
      const double uz_p = w0 + u0 * ty - v0 * tx;
      ux[ip] = u0 + uy_p * sz - uz_p * sy + econst * Ex[ip];
      uy[ip] = v0 + uz_p * sx - ux_p * sz + econst * Ey[ip];
      uz[ip] = w0 + ux_p * sy - uy_p * sx + econst * Ez[ip];
    }
  }
}
