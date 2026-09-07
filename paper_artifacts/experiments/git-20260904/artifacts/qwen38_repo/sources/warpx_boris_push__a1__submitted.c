// Optimized WarpX Boris momentum pusher.
//
// The naive implementation is a straight NumPy-to-C lowering: ~20 separate
// passes over the particle arrays plus 12 temporary arrays (malloc'd and
// freed on every call).  The per-particle math is a pure elementwise map
// with no cross-lane dependence, so the whole update collapses to ONE fused
// pass over the particles, split across OpenMP threads.
//
// Numerics: the floating-point operation ORDER inside a particle is kept
// exactly as in the reference lowering (same associativity, same
// mul/add/div/sqrt sequence, same FMA-formation opportunities), so the
// results are bit-identical to the naive implementation -- and hence to
// the NumPy reference within the same tolerance the baseline passes.
// No reassociation and no math changes.
//
// Implementation notes:
//  * With this toolchain the vectorizer refuses to vectorize loops that live
//    directly inside an OpenMP parallel region, so the per-thread work is
//    done in noinline static helpers (one plain, fully vectorizable loop
//    each); the parallel region only partitions the index range and
//    dispatches.
//  * The helpers take the particle arrays as individual restrict pointers:
//    reading them through one struct pointer would force aliasing-versioned
//    code paths and lose vectorization for most of the push types.

#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif
#ifdef I
#undef I
#endif

constexpr double dt = 1e-13;

/* Full push: first E half, rotation (no t rescale), second E half.  mpt == 0. */
static void __attribute__((noinline)) core_full(
    const double *restrict Bx, const double *restrict By, const double *restrict Bz,
    const double *restrict Ex, const double *restrict Ey, const double *restrict Ez,
    double *restrict ux, double *restrict uy, double *restrict uz,
    const double econst, const int64_t lo, const int64_t hi) {
    const double inv_c2 = 1.1126500560536185e-17;
    for (int64_t i = lo; i < hi; ++i) {
        double ux0 = (ux[i] + (econst * Ex[i]));
        double uy0 = (uy[i] + (econst * Ey[i]));
        double uz0 = (uz[i] + (econst * Ez[i]));
        double cb1 = sqrt((1.0 + ((((ux0 * ux0) + (uy0 * uy0)) + (uz0 * uz0)) * inv_c2)));
        double inv_gamma = (1.0 / cb1);
        double t_base = (econst * inv_gamma);
        double tx = (t_base * Bx[i]);
        double ty = (t_base * By[i]);
        double tz = (t_base * Bz[i]);
        double tsqi = (2.0 / (((1.0 + (tx * tx)) + (ty * ty)) + (tz * tz)));
        double sx = (tx * tsqi);
        double sy = (ty * tsqi);
        double sz = (tz * tsqi);
        double ux_p = ((ux0 + (uy0 * tz)) - (uz0 * ty));
        double uy_p = ((uy0 + (uz0 * tx)) - (ux0 * tz));
        double uz_p = ((uz0 + (ux0 * ty)) - (uy0 * tx));
        double nx = ux0 + ((uy_p * sz) - (uz_p * sy)) + (econst * Ex[i]);
        double ny = uy0 + ((uz_p * sx) - (ux_p * sz)) + (econst * Ey[i]);
        double nz = uz0 + ((ux_p * sy) - (uy_p * sx)) + (econst * Ez[i]);
        ux[i] = nx;
        uy[i] = ny;
        uz[i] = nz;
    }
}

/* First half push: first E half, rotation with t rescale.  mpt == 1. */
static void __attribute__((noinline)) core_first(
    const double *restrict Bx, const double *restrict By, const double *restrict Bz,
    const double *restrict Ex, const double *restrict Ey, const double *restrict Ez,
    double *restrict ux, double *restrict uy, double *restrict uz,
    const double econst, const int64_t lo, const int64_t hi) {
    const double inv_c2 = 1.1126500560536185e-17;
    for (int64_t i = lo; i < hi; ++i) {
        double ux0 = (ux[i] + (econst * Ex[i]));
        double uy0 = (uy[i] + (econst * Ey[i]));
        double uz0 = (uz[i] + (econst * Ez[i]));
        double cb1 = sqrt((1.0 + ((((ux0 * ux0) + (uy0 * uy0)) + (uz0 * uz0)) * inv_c2)));
        double inv_gamma = (1.0 / cb1);
        double t_base = (econst * inv_gamma);
        double tx = (t_base * Bx[i]);
        double ty = (t_base * By[i]);
        double tz = (t_base * Bz[i]);
        double tsq = (((tx * tx) + (ty * ty)) + (tz * tz));
        bool hf = (tsq > 0.0);
        double safe_tsq = (hf ? tsq : 1.0);
        double factor = (hf ? ((sqrt((1.0 + tsq)) - 1.0) / safe_tsq) : 0.5);
        tx *= factor;
        ty *= factor;
        tz *= factor;
        double tsqi = (2.0 / (((1.0 + (tx * tx)) + (ty * ty)) + (tz * tz)));
        double sx = (tx * tsqi);
        double sy = (ty * tsqi);
        double sz = (tz * tsqi);
        double ux_p = ((ux0 + (uy0 * tz)) - (uz0 * ty));
        double uy_p = ((uy0 + (uz0 * tx)) - (ux0 * tz));
        double uz_p = ((uz0 + (ux0 * ty)) - (uy0 * tx));
        double nx = ux0 + ((uy_p * sz) - (uz_p * sy));
        double ny = uy0 + ((uz_p * sx) - (ux_p * sz));
        double nz = uz0 + ((ux_p * sy) - (uy_p * sx));
        ux[i] = nx;
        uy[i] = ny;
        uz[i] = nz;
    }
}

/* Second half push: rotation with t rescale, second E half.  mpt == 2. */
static void __attribute__((noinline)) core_second(
    const double *restrict Bx, const double *restrict By, const double *restrict Bz,
    const double *restrict Ex, const double *restrict Ey, const double *restrict Ez,
    double *restrict ux, double *restrict uy, double *restrict uz,
    const double econst, const int64_t lo, const int64_t hi) {
    const double inv_c2 = 1.1126500560536185e-17;
    for (int64_t i = lo; i < hi; ++i) {
        double ux0 = ux[i];
        double uy0 = uy[i];
        double uz0 = uz[i];
        double cb1 = sqrt((1.0 + ((((ux0 * ux0) + (uy0 * uy0)) + (uz0 * uz0)) * inv_c2)));
        double inv_gamma = (1.0 / cb1);
        double t_base = (econst * inv_gamma);
        double tx = (t_base * Bx[i]);
        double ty = (t_base * By[i]);
        double tz = (t_base * Bz[i]);
        double tsq = (((tx * tx) + (ty * ty)) + (tz * tz));
        bool hf = (tsq > 0.0);
        double safe_tsq = (hf ? tsq : 1.0);
        double factor = (hf ? ((sqrt((1.0 + tsq)) - 1.0) / safe_tsq) : 0.5);
        tx *= factor;
        ty *= factor;
        tz *= factor;
        double tsqi = (2.0 / (((1.0 + (tx * tx)) + (ty * ty)) + (tz * tz)));
        double sx = (tx * tsqi);
        double sy = (ty * tsqi);
        double sz = (tz * tsqi);
        double ux_p = ((ux0 + (uy0 * tz)) - (uz0 * ty));
        double uy_p = ((uy0 + (uz0 * tx)) - (ux0 * tz));
        double uz_p = ((uz0 + (ux0 * ty)) - (uy0 * tx));
        double nx = ux0 + ((uy_p * sz) - (uz_p * sy)) + (econst * Ex[i]);
        double ny = uy0 + ((uz_p * sx) - (ux_p * sz)) + (econst * Ey[i]);
        double nz = uz0 + ((ux_p * sy) - (uy_p * sx)) + (econst * Ez[i]);
        ux[i] = nx;
        uy[i] = ny;
        uz[i] = nz;
    }
}

/* Any other momentum_push_type: plain rotation without E pushes or rescale,
   exactly what the reference does for mpt not in {0,1,2}. */
static void __attribute__((noinline)) core_rot(
    const double *restrict Bx, const double *restrict By, const double *restrict Bz,
    const double *restrict Ex, const double *restrict Ey, const double *restrict Ez,
    double *restrict ux, double *restrict uy, double *restrict uz,
    const double econst, const int64_t lo, const int64_t hi) {
    (void)Ex; (void)Ey; (void)Ez;
    const double inv_c2 = 1.1126500560536185e-17;
    for (int64_t i = lo; i < hi; ++i) {
        double ux0 = ux[i];
        double uy0 = uy[i];
        double uz0 = uz[i];
        double cb1 = sqrt((1.0 + ((((ux0 * ux0) + (uy0 * uy0)) + (uz0 * uz0)) * inv_c2)));
        double inv_gamma = (1.0 / cb1);
        double t_base = (econst * inv_gamma);
        double tx = (t_base * Bx[i]);
        double ty = (t_base * By[i]);
        double tz = (t_base * Bz[i]);
        double tsqi = (2.0 / (((1.0 + (tx * tx)) + (ty * ty)) + (tz * tz)));
        double sx = (tx * tsqi);
        double sy = (ty * tsqi);
        double sz = (tz * tsqi);
        double ux_p = ((ux0 + (uy0 * tz)) - (uz0 * ty));
        double uy_p = ((uy0 + (uz0 * tx)) - (ux0 * tz));
        double uz_p = ((uz0 + (ux0 * ty)) - (uy0 * tx));
        double nx = ux0 + ((uy_p * sz) - (uz_p * sy));
        double ny = uy0 + ((uz_p * sx) - (ux_p * sz));
        double nz = uz0 + ((ux_p * sy) - (uy_p * sx));
        ux[i] = nx;
        uy[i] = ny;
        uz[i] = nz;
    }
}

void warpx_boris_push_fp64(const double *restrict Bx, const double *restrict By, const double *restrict Bz, const double *restrict Ex, const double *restrict Ey, const double *restrict Ez, double *restrict ux, double *restrict uy, double *restrict uz, const double m, const int64_t momentum_push_type, const int64_t np_particles, const double q) {
        int64_t mpt;
        double __inl1_econst;
        mpt = ((int64_t)(momentum_push_type));
        __inl1_econst = (((0.5 * q) * dt) / m);
        if (np_particles <= 0) return;

        #pragma omp parallel
        {
          int64_t nt = omp_get_num_threads();
          int64_t tid = omp_get_thread_num();
          int64_t chunk = (np_particles + nt - 1) / nt;
          int64_t lo = tid * chunk;
          int64_t hi = (lo + chunk < np_particles) ? (lo + chunk) : np_particles;
          if (lo < hi) {
            if (mpt == 0)            core_full(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, __inl1_econst, lo, hi);
            else if (mpt == 1)       core_first(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, __inl1_econst, lo, hi);
            else if (mpt == 2)       core_second(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, __inl1_econst, lo, hi);
            else                     core_rot(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, __inl1_econst, lo, hi);
          }
        }
}
