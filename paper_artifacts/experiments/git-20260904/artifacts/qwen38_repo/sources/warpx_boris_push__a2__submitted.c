// Optimized WarpX Boris momentum pusher (in-place, single fused pass).
//
// The naive seed did ~25 sequential passes over the particle arrays plus 13
// malloc'd temporaries.  The per-particle update is an embarrassingly
// parallel elementwise map, so this version:
//   * fuses everything into ONE pass with zero temporaries (pure scalar
//     locals per particle),
//   * splits into four code paths by momentum_push_type so every runtime
//     branch becomes a compile-time constant and the hot loop vectorizes,
//   * runs the loop under OpenMP (falling back to a serial vectorized loop
//     for small n where thread-spawn overhead would dominate).
//
// Every arithmetic expression keeps the reference parenthesization, so the
// results stay bit-compatible with the NumPy oracle.

#include <stdint.h>
#include <math.h>

#define WBP_INV_C2 1.1126500560536185e-17 /* 1 / c^2, c = 299792458 m/s */

#define WBP_BODY(_ID, _E1, _RES, _E2, _OMP)                                     \
  static void _warpx_push_##_ID(const double *restrict Bx,                      \
                                const double *restrict By,                      \
                                const double *restrict Bz,                      \
                                const double *restrict Ex,                      \
                                const double *restrict Ey,                      \
                                const double *restrict Ez,                      \
                                double *restrict ux,                            \
                                double *restrict uy,                            \
                                double *restrict uz,                            \
                                double econst, int64_t n)                       \
  {                                                                             \
    _OMP("omp parallel for schedule(static)")                                   \
    for (int64_t i = 0; i < n; ++i) {                                           \
      double ux_i = ux[i];                                                      \
      double uy_i = uy[i];                                                      \
      double uz_i = uz[i];                                                      \
      if (_E1) {                                                                \
        ux_i += econst * Ex[i];                                                 \
        uy_i += econst * Ey[i];                                                 \
        uz_i += econst * Ez[i];                                                 \
      }                                                                         \
      const double cb1 =                                                        \
          sqrt(1.0 + (((ux_i * ux_i) + (uy_i * uy_i)) + (uz_i * uz_i)) *        \
                     WBP_INV_C2);                                               \
      const double ig = 1.0 / cb1;                                              \
      double tx = (econst * ig) * Bx[i];                                        \
      double ty = (econst * ig) * By[i];                                        \
      double tz = (econst * ig) * Bz[i];                                        \
      if (_RES) {                                                               \
        const double tsq = ((tx * tx) + (ty * ty)) + (tz * tz);                 \
        const double factor =                                                   \
            (tsq > 0.0) ? ((sqrt(1.0 + tsq) - 1.0) / tsq) : 0.5;                \
        tx *= factor;                                                           \
        ty *= factor;                                                           \
        tz *= factor;                                                           \
      }                                                                         \
      const double tsqi =                                                       \
          2.0 / (((1.0 + (tx * tx)) + (ty * ty)) + (tz * tz));                  \
      const double sx = tx * tsqi;                                              \
      const double sy = ty * tsqi;                                              \
      const double sz = tz * tsqi;                                              \
      const double uxp = (ux_i + (uy_i * tz)) - (uz_i * ty);                    \
      const double uyp = (uy_i + (uz_i * tx)) - (ux_i * tz);                    \
      const double uzp = (uz_i + (ux_i * ty)) - (uy_i * tx);                    \
      ux_i += (uyp * sz) - (uzp * sy);                                          \
      uy_i += (uzp * sx) - (uxp * sz);                                          \
      uz_i += (uxp * sy) - (uyp * sx);                                          \
      if (_E2) {                                                                \
        ux_i += econst * Ex[i];                                                 \
        uy_i += econst * Ey[i];                                                 \
        uz_i += econst * Ez[i];                                                 \
      }                                                                         \
      ux[i] = ux_i;                                                             \
      uy[i] = uy_i;                                                             \
      uz[i] = uz_i;                                                             \
    }                                                                           \
  }

#define WBP_OMP(...) _Pragma(__VA_ARGS__)
#define WBP_NOOP(...)

WBP_BODY(full_omp,   1, 0, 1, WBP_OMP)
WBP_BODY(first_omp,  1, 1, 0, WBP_OMP)
WBP_BODY(second_omp, 0, 1, 1, WBP_OMP)
WBP_BODY(bonly_omp,  0, 0, 0, WBP_OMP)
WBP_BODY(full_ser,   1, 0, 1, WBP_NOOP)
WBP_BODY(first_ser,  1, 1, 0, WBP_NOOP)
WBP_BODY(second_ser, 0, 1, 1, WBP_NOOP)
WBP_BODY(bonly_ser,  0, 0, 0, WBP_NOOP)

/* Below this many particles the OpenMP spawn+barrier is not worth it. */
#define WBP_OMP_MIN_N 131072

void warpx_boris_push_fp64(const double *restrict Bx,
                           const double *restrict By,
                           const double *restrict Bz,
                           const double *restrict Ex,
                           const double *restrict Ey,
                           const double *restrict Ez,
                           double *restrict ux,
                           double *restrict uy,
                           double *restrict uz,
                           const double m,
                           const int64_t momentum_push_type,
                           const int64_t np_particles,
                           const double q) {
  if (np_particles <= 0) return;
  const double dt = 1e-13;
  const double econst = ((0.5 * q) * dt) / m;
  if (np_particles < WBP_OMP_MIN_N) {
    switch (momentum_push_type) {
      case 0:  _warpx_push_full_ser(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, econst, np_particles); return;
      case 1:  _warpx_push_first_ser(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, econst, np_particles); return;
      case 2:  _warpx_push_second_ser(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, econst, np_particles); return;
      default: _warpx_push_bonly_ser(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, econst, np_particles); return;
    }
  }
  switch (momentum_push_type) {
    case 0:  _warpx_push_full_omp(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, econst, np_particles); return;
    case 1:  _warpx_push_first_omp(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, econst, np_particles); return;
    case 2:  _warpx_push_second_omp(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, econst, np_particles); return;
    default: _warpx_push_bonly_omp(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, econst, np_particles); return;
  }
}
