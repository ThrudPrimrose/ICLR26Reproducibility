/* Optimized WarpX Boris particle-momentum pusher.
 *
 * Single fused pass per particle (no temp arrays, no repeated array sweeps),
 * compile-time specialized per momentum_push_type, AVX-512 vectorized with a
 * 3x manual unroll so three independent sqrt/div dependency chains are in
 * flight (the auto-vectorizer alone keeps one chain, which serializes the
 * sqrt+div latency). OpenMP-parallel across particles above a size threshold
 * where the team setup pays off.
 *
 * Arithmetic order matches the NumPy reference exactly and FMA contraction is
 * disabled so the result is bit-identical to the no-FMA NumPy oracle for any
 * input magnitude.
 */
#include <stdint.h>
#include <math.h>
#include <omp.h>

/* Keep the FP semantics bit-identical to the NumPy oracle: never fuse a*b+c
 * into an FMA (FMA rounds once where NumPy's separate mul+add round twice). */
#pragma GCC optimize("fp-contract=off")

#define C_LIGHT 299792458.0
#define INV_C2 (1.0 / (C_LIGHT * C_LIGHT))
#define DT 1e-13

/* One particle's Boris push. ip indexes every array. P1/H/P2 select the
 * first E half-push, the half-rotation t-rescale, and the second E half-push.
 * Each line mirrors one NumPy statement, in the same left-to-right order. */
#define BORIS_BODY(ip, P1, H, P2)                                       \
do {                                                                    \
  double ux_ = ux[ip], uy_ = uy[ip], uz_ = uz[ip];                      \
  double Ex_ = Ex[ip], Ey_ = Ey[ip], Ez_ = Ez[ip];                      \
  double Bx_ = Bx[ip], By_ = By[ip], Bz_ = Bz[ip];                      \
                                                                        \
  if (P1) {                                                             \
    ux_ += econst * Ex_;                                                \
    uy_ += econst * Ey_;                                                \
    uz_ += econst * Ez_;                                                \
  }                                                                     \
                                                                        \
  double s = (ux_ * ux_ + uy_ * uy_) + uz_ * uz_;                       \
  double inv_gamma = 1.0 / sqrt(1.0 + s * inv_c2);                      \
                                                                        \
  double cg = econst * inv_gamma;                                       \
  double tx = cg * Bx_, ty = cg * By_, tz = cg * Bz_;                   \
                                                                        \
  if (H) {                                                              \
    double tsq = (tx * tx + ty * ty) + tz * tz;                         \
    double safe_tsq = (tsq > 0.0) ? tsq : 1.0;                          \
    double ratio = (sqrt(1.0 + tsq) - 1.0) / safe_tsq;                  \
    double factor = (tsq > 0.0) ? ratio : 0.5;                          \
    tx *= factor;                                                       \
    ty *= factor;                                                       \
    tz *= factor;                                                       \
  }                                                                     \
                                                                        \
  double tsqi = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz);              \
  double sx = tx * tsqi, sy = ty * tsqi, sz = tz * tsqi;                \
                                                                        \
  double ux_p = ux_ + uy_ * tz - uz_ * ty;                              \
  double uy_p = uy_ + uz_ * tx - ux_ * tz;                              \
  double uz_p = uz_ + ux_ * ty - uy_ * tx;                              \
                                                                        \
  double nux = ux_ + (uy_p * sz - uz_p * sy);                           \
  double nuy = uy_ + (uz_p * sx - ux_p * sz);                           \
  double nuz = uz_ + (ux_p * sy - uy_p * sx);                           \
                                                                        \
  if (P2) {                                                             \
    nux += econst * Ex_;                                                \
    nuy += econst * Ey_;                                                \
    nuz += econst * Ez_;                                                \
  }                                                                     \
                                                                        \
  ux[ip] = nux;                                                         \
  uy[ip] = nuy;                                                         \
  uz[ip] = nuz;                                                         \
} while (0)

typedef const double *restrict farr;
typedef double *restrict oarr;

/* Dispatch on size: below SERIAL_THRESH the OpenMP team barrier/spin overhead
 * exceeds the work, so run a single-thread vectorized loop; above it the
 * parallel loop wins. Both branches keep the exact arithmetic (bit-identical).
 *
 * The vectorized loop runs 3 independent particles per iteration (ip, ip+1,
 * ip+2): 3-way unroll is the largest factor GCC will still vectorize here,
 * and it keeps three sqrt/div latency chains in flight. The <3-particle tail
 * is a trivial second vectorized loop (serial branch) or an inline master
 * pass after the team (parallel branch).
 */
#define SERIAL_THRESH 8192
#define KERNEL(NAME, P1, H, P2)                                              \
static void NAME(farr Bx, farr By, farr Bz, farr Ex, farr Ey, farr Ez,       \
                 oarr ux, oarr uy, oarr uz, double econst, double inv_c2, long np) { \
  const long n3 = np - (np % 3);                                             \
  if (np <= SERIAL_THRESH) {                                                 \
    _Pragma("omp simd")                                                      \
    for (long ip = 0; ip < n3; ip += 3) {                                    \
      BORIS_BODY(ip, P1, H, P2); BORIS_BODY(ip + 1, P1, H, P2);              \
      BORIS_BODY(ip + 2, P1, H, P2);                                         \
    }                                                                        \
    _Pragma("omp simd")                                                      \
    for (long ip = n3; ip < np; ++ip) { BORIS_BODY(ip, P1, H, P2); }         \
  } else {                                                                   \
    _Pragma("omp parallel for simd schedule(static)")                        \
    for (long ip = 0; ip < n3; ip += 3) {                                    \
      BORIS_BODY(ip, P1, H, P2); BORIS_BODY(ip + 1, P1, H, P2);              \
      BORIS_BODY(ip + 2, P1, H, P2);                                         \
    }                                                                        \
    for (long ip = n3; ip < np; ++ip) { BORIS_BODY(ip, P1, H, P2); }         \
  }                                                                          \
}
KERNEL(k_full, 1, 0, 1)
KERNEL(k_first, 1, 1, 0)
KERNEL(k_second, 0, 1, 1)
KERNEL(k_none, 0, 0, 0)

void warpx_boris_push_fp64(const double *restrict Bx, const double *restrict By,
                           const double *restrict Bz, const double *restrict Ex,
                           const double *restrict Ey, const double *restrict Ez,
                           double *restrict ux, double *restrict uy, double *restrict uz,
                           const double m, const int64_t momentum_push_type,
                           const int64_t np_particles, const double q) {
  static int nt_once = 1;
  if (nt_once) { nt_once = 0; omp_set_num_threads(9); }
  const long np = (long)np_particles;
  const int64_t mpt = momentum_push_type;
  const double econst = (0.5 * q * DT) / m;
  const double inv_c2 = INV_C2;

  if (mpt == 0)        k_full(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, econst, inv_c2, np);
  else if (mpt == 1)   k_first(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, econst, inv_c2, np);
  else if (mpt == 2)   k_second(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, econst, inv_c2, np);
  else                 k_none(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, econst, inv_c2, np);
}
