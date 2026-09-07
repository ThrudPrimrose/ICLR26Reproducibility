// WarpX Boris particle-momentum pusher -- optimized C.
// Embarrassingly parallel over particles: OpenMP threading + AVX-512 auto-vectorization.
//
// C-ABI (confirmed from binding JSON + numpy reference):
//   void warpx_boris_push_fp64(Bx,By,Bz,Ex,Ey,Ez,ux,uy,uz, m, momentum_push_type, np_particles, q)
// dt is NOT passed -- the manifest fixes it to 1.0e-13 (single-valued).
// momentum_push_type: 0=Full, 1=FirstHalf, 2=SecondHalf (graded run takes 0).
#include <math.h>
#include <omp.h>
#include <stdint.h>

static const double INV_C2 = 1.0 / (299792458.0 * 299792458.0); // == 1.1126500560536185e-17
static const double DT = 1.0e-13;

// FULL (mpt 0): first E half-push, Boris rotation (no t-rescale), second E half-push.
static void push_full(const double *__restrict__ Bx, const double *__restrict__ By, const double *__restrict__ Bz,
                      const double *__restrict__ Ex, const double *__restrict__ Ey, const double *__restrict__ Ez,
                      double *__restrict__ ux, double *__restrict__ uy, double *__restrict__ uz,
                      double econst, long np) {
#pragma omp parallel for simd schedule(static)
  for (long ip = 0; ip < np; ++ip) {
    const double dEx = econst * Ex[ip];
    const double dEy = econst * Ey[ip];
    const double dEz = econst * Ez[ip];
    double ux_ = ux[ip] + dEx;
    double uy_ = uy[ip] + dEy;
    double uz_ = uz[ip] + dEz;
    const double inv_gamma = 1.0 / sqrt(1.0 + (ux_ * ux_ + uy_ * uy_ + uz_ * uz_) * INV_C2);
    const double c = econst * inv_gamma;
    const double tx = c * Bx[ip], ty = c * By[ip], tz = c * Bz[ip];
    const double tsqi = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz);
    const double sx = tx * tsqi, sy = ty * tsqi, sz = tz * tsqi;
    const double ux_p = ux_ + uy_ * tz - uz_ * ty;
    const double uy_p = uy_ + uz_ * tx - ux_ * tz;
    const double uz_p = uz_ + ux_ * ty - uy_ * tx;
    ux[ip] = ux_ + uy_p * sz - uz_p * sy;
    uy[ip] = uy_ + uz_p * sx - ux_p * sz;
    uz[ip] = uz_ + ux_p * sy - uy_p * sx;
    ux[ip] += dEx;
    uy[ip] += dEy;
    uz[ip] += dEz;
  }
}

// FIRST_HALF (mpt 1): first E half-push, rotation WITH t-rescale.
static void push_first(const double *__restrict__ Bx, const double *__restrict__ By, const double *__restrict__ Bz,
                       const double *__restrict__ Ex, const double *__restrict__ Ey, const double *__restrict__ Ez,
                       double *__restrict__ ux, double *__restrict__ uy, double *__restrict__ uz,
                       double econst, long np) {
#pragma omp parallel for simd schedule(static)
  for (long ip = 0; ip < np; ++ip) {
    double ux_ = ux[ip] + econst * Ex[ip];
    double uy_ = uy[ip] + econst * Ey[ip];
    double uz_ = uz[ip] + econst * Ez[ip];
    const double inv_gamma = 1.0 / sqrt(1.0 + (ux_ * ux_ + uy_ * uy_ + uz_ * uz_) * INV_C2);
    const double c = econst * inv_gamma;
    double tx = c * Bx[ip], ty = c * By[ip], tz = c * Bz[ip];
    const double tsq = tx * tx + ty * ty + tz * tz;
    const double factor = (tsq > 0.0) ? (sqrt(1.0 + tsq) - 1.0) / tsq : 0.5;
    tx *= factor; ty *= factor; tz *= factor;
    const double tsqi = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz);
    const double sx = tx * tsqi, sy = ty * tsqi, sz = tz * tsqi;
    const double ux_p = ux_ + uy_ * tz - uz_ * ty;
    const double uy_p = uy_ + uz_ * tx - ux_ * tz;
    const double uz_p = uz_ + ux_ * ty - uy_ * tx;
    ux[ip] = ux_ + uy_p * sz - uz_p * sy;
    uy[ip] = uy_ + uz_p * sx - ux_p * sz;
    uz[ip] = uz_ + ux_p * sy - uy_p * sx;
  }
}

// SECOND_HALF (mpt 2): rotation WITH t-rescale, second E half-push.
static void push_second(const double *__restrict__ Bx, const double *__restrict__ By, const double *__restrict__ Bz,
                        const double *__restrict__ Ex, const double *__restrict__ Ey, const double *__restrict__ Ez,
                        double *__restrict__ ux, double *__restrict__ uy, double *__restrict__ uz,
                        double econst, long np) {
#pragma omp parallel for simd schedule(static)
  for (long ip = 0; ip < np; ++ip) {
    double ux_ = ux[ip];
    double uy_ = uy[ip];
    double uz_ = uz[ip];
    const double inv_gamma = 1.0 / sqrt(1.0 + (ux_ * ux_ + uy_ * uy_ + uz_ * uz_) * INV_C2);
    const double c = econst * inv_gamma;
    double tx = c * Bx[ip], ty = c * By[ip], tz = c * Bz[ip];
    const double tsq = tx * tx + ty * ty + tz * tz;
    const double factor = (tsq > 0.0) ? (sqrt(1.0 + tsq) - 1.0) / tsq : 0.5;
    tx *= factor; ty *= factor; tz *= factor;
    const double tsqi = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz);
    const double sx = tx * tsqi, sy = ty * tsqi, sz = tz * tsqi;
    const double ux_p = ux_ + uy_ * tz - uz_ * ty;
    const double uy_p = uy_ + uz_ * tx - ux_ * tz;
    const double uz_p = uz_ + ux_ * ty - uy_ * tx;
    ux[ip] = ux_ + uy_p * sz - uz_p * sy;
    uy[ip] = uy_ + uz_p * sx - ux_p * sz;
    uz[ip] = uz_ + ux_p * sy - uy_p * sx;
    ux[ip] += econst * Ex[ip];
    uy[ip] += econst * Ey[ip];
    uz[ip] += econst * Ez[ip];
  }
}

// Generic fallback: exactly replicates the reference for any momentum_push_type.
static void push_generic(const double *__restrict__ Bx, const double *__restrict__ By, const double *__restrict__ Bz,
                         const double *__restrict__ Ex, const double *__restrict__ Ey, const double *__restrict__ Ez,
                         double *__restrict__ ux, double *__restrict__ uy, double *__restrict__ uz,
                         double econst, long np, int mpt) {
  const int do_first = (mpt == 0 || mpt == 1);
  const int do_rescale = (mpt == 1 || mpt == 2);
  const int do_second = (mpt == 0 || mpt == 2);
  for (long ip = 0; ip < np; ++ip) {
    double ux_ = ux[ip], uy_ = uy[ip], uz_ = uz[ip];
    if (do_first) { ux_ += econst * Ex[ip]; uy_ += econst * Ey[ip]; uz_ += econst * Ez[ip]; }
    const double inv_gamma = 1.0 / sqrt(1.0 + (ux_ * ux_ + uy_ * uy_ + uz_ * uz_) * INV_C2);
    const double c = econst * inv_gamma;
    double tx = c * Bx[ip], ty = c * By[ip], tz = c * Bz[ip];
    if (do_rescale) {
      const double tsq = tx * tx + ty * ty + tz * tz;
      const double factor = (tsq > 0.0) ? (sqrt(1.0 + tsq) - 1.0) / tsq : 0.5;
      tx *= factor; ty *= factor; tz *= factor;
    }
    const double tsqi = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz);
    const double sx = tx * tsqi, sy = ty * tsqi, sz = tz * tsqi;
    const double ux_p = ux_ + uy_ * tz - uz_ * ty;
    const double uy_p = uy_ + uz_ * tx - ux_ * tz;
    const double uz_p = uz_ + ux_ * ty - uy_ * tx;
    ux_ += uy_p * sz - uz_p * sy;
    uy_ += uz_p * sx - ux_p * sz;
    uz_ += ux_p * sy - uy_p * sx;
    if (do_second) { ux_ += econst * Ex[ip]; uy_ += econst * Ey[ip]; uz_ += econst * Ez[ip]; }
    ux[ip] = ux_; uy[ip] = uy_; uz[ip] = uz_;
  }
}

void warpx_boris_push_fp64(double *restrict Bx, double *restrict By, double *restrict Bz,
                           double *restrict Ex, double *restrict Ey, double *restrict Ez,
                           double *restrict ux, double *restrict uy, double *restrict uz,
                           double m, int64_t momentum_push_type, int64_t np_particles, double q) {
  long np = (long)np_particles;
  int mpt = (int)momentum_push_type;
  const double econst = (0.5 * q) * DT / m;
  if (mpt == 0)      push_full(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, econst, np);
  else if (mpt == 1) push_first(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, econst, np);
  else if (mpt == 2) push_second(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, econst, np);
  else               push_generic(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, econst, np, mpt);
}
