#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>

#define WARPX_OMP _Pragma("omp parallel for schedule(static)")
static const double C_LIGHT = 299792458.0;
static const double INV_C2 = 1.0 / (C_LIGHT * C_LIGHT);

#if defined(__AVX512F__)
/* Generic vectorized body for 8 particles.
   e1: do first E half-push; e2: do second E half-push; rescale: apply t rescale.
   Called with literal constants so GCC constant-folds the branches per call. */
static inline void push8(const double *__restrict__ Bx, const double *__restrict__ By,
                         const double *__restrict__ Bz, const double *__restrict__ Ex,
                         const double *__restrict__ Ey, const double *__restrict__ Ez,
                         double *__restrict__ ux, double *__restrict__ uy,
                         double *__restrict__ uz, long base,
                         double econst, int e1, int e2, int rescale)
{
    __m512d vBx = _mm512_load_pd(Bx + base);
    __m512d vBy = _mm512_load_pd(By + base);
    __m512d vBz = _mm512_load_pd(Bz + base);
    __m512d vEx = _mm512_load_pd(Ex + base);
    __m512d vEy = _mm512_load_pd(Ey + base);
    __m512d vEz = _mm512_load_pd(Ez + base);
    __m512d vux = _mm512_load_pd(ux + base);
    __m512d vuy = _mm512_load_pd(uy + base);
    __m512d vuz = _mm512_load_pd(uz + base);

    __m512d ve   = _mm512_set1_pd(econst);
    __m512d vone = _mm512_set1_pd(1.0);
    __m512d vtwo = _mm512_set1_pd(2.0);
    __m512d vc   = _mm512_set1_pd(INV_C2);

    if (e1) {
        vux = _mm512_fmadd_pd(ve, vEx, vux);
        vuy = _mm512_fmadd_pd(ve, vEy, vuy);
        vuz = _mm512_fmadd_pd(ve, vEz, vuz);
    }

    __m512d s = _mm512_mul_pd(vux, vux);
    s = _mm512_fmadd_pd(vuy, vuy, s);
    s = _mm512_fmadd_pd(vuz, vuz, s);
    s = _mm512_fmadd_pd(s, vc, vone);                 /* 1 + u2*inv_c2 */
    __m512d ig = _mm512_div_pd(vone, _mm512_sqrt_pd(s));

    __m512d tc = _mm512_mul_pd(ve, ig);
    __m512d tx = _mm512_mul_pd(tc, vBx);
    __m512d ty = _mm512_mul_pd(tc, vBy);
    __m512d tz = _mm512_mul_pd(tc, vBz);

    if (rescale) {
        __m512d tsq = _mm512_mul_pd(tx, tx);
        tsq = _mm512_fmadd_pd(ty, ty, tsq);
        tsq = _mm512_fmadd_pd(tz, tz, tsq);
        __mmask8 has = _mm512_cmp_pd_mask(tsq, _mm512_setzero_pd(), _CMP_GT_OQ);
        __m512d safe = _mm512_mask_blend_pd(has, vone, tsq);   /* has? tsq : 1.0 */
        __m512d fac = _mm512_div_pd(_mm512_sub_pd(_mm512_sqrt_pd(_mm512_add_pd(vone, tsq)), vone), safe);
        fac = _mm512_mask_blend_pd(has, _mm512_set1_pd(0.5), fac); /* has? fac : 0.5 */
        tx = _mm512_mul_pd(tx, fac);
        ty = _mm512_mul_pd(ty, fac);
        tz = _mm512_mul_pd(tz, fac);
    }

    __m512d t2 = _mm512_mul_pd(tx, tx);
    t2 = _mm512_fmadd_pd(ty, ty, t2);
    t2 = _mm512_fmadd_pd(tz, tz, t2);
    __m512d tsqi = _mm512_div_pd(vtwo, _mm512_add_pd(vone, t2));
    __m512d sx = _mm512_mul_pd(tx, tsqi);
    __m512d sy = _mm512_mul_pd(ty, tsqi);
    __m512d sz = _mm512_mul_pd(tz, tsqi);

    __m512d ux_p = _mm512_fmadd_pd(vuy, tz, vux);
    ux_p = _mm512_fnmadd_pd(vuz, ty, ux_p);
    __m512d uy_p = _mm512_fmadd_pd(vuz, tx, vuy);
    uy_p = _mm512_fnmadd_pd(vux, tz, uy_p);
    __m512d uz_p = _mm512_fmadd_pd(vux, ty, vuz);
    uz_p = _mm512_fnmadd_pd(vuy, tx, uz_p);

    vux = _mm512_fmadd_pd(uy_p, sz, vux);
    vux = _mm512_fnmadd_pd(uz_p, sy, vux);
    vuy = _mm512_fmadd_pd(uz_p, sx, vuy);
    vuy = _mm512_fnmadd_pd(ux_p, sz, vuy);
    vuz = _mm512_fmadd_pd(ux_p, sy, vuz);
    vuz = _mm512_fnmadd_pd(uy_p, sx, vuz);

    if (e2) {
        vux = _mm512_fmadd_pd(ve, vEx, vux);
        vuy = _mm512_fmadd_pd(ve, vEy, vuy);
        vuz = _mm512_fmadd_pd(ve, vEz, vuz);
    }

    _mm512_store_pd(ux + base, vux);
    _mm512_store_pd(uy + base, vuy);
    _mm512_store_pd(uz + base, vuz);
}

#endif

/* Scalar fallback for the tail / no-AVX512. */
static inline void push1(const double *Bx, const double *By, const double *Bz,
                         const double *Ex, const double *Ey, const double *Ez,
                         double *ux, double *uy, double *uz, long ip,
                         double econst, int e1, int e2, int rescale)
{
    double uxv = ux[ip], uyv = uy[ip], uzv = uz[ip];
    if (e1) { uxv += econst*Ex[ip]; uyv += econst*Ey[ip]; uzv += econst*Ez[ip]; }
    double ig = 1.0 / sqrt(1.0 + (uxv*uxv + uyv*uyv + uzv*uzv) * INV_C2);
    double tx = econst*ig*Bx[ip], ty = econst*ig*By[ip], tz = econst*ig*Bz[ip];
    if (rescale) {
        double tsq = tx*tx + ty*ty + tz*tz;
        double fac = (tsq > 0.0) ? (sqrt(1.0 + tsq) - 1.0) / tsq : 0.5;
        tx *= fac; ty *= fac; tz *= fac;
    }
    double tsqi = 2.0 / (1.0 + tx*tx + ty*ty + tz*tz);
    double sx = tx*tsqi, sy = ty*tsqi, sz = tz*tsqi;
    double ux_p = uxv + uyv*tz - uzv*ty;
    double uy_p = uyv + uzv*tx - uxv*tz;
    double uz_p = uzv + uxv*ty - uyv*tx;
    uxv += uy_p*sz - uz_p*sy;
    uyv += uz_p*sx - ux_p*sz;
    uzv += ux_p*sy - uy_p*sx;
    if (e2) { uxv += econst*Ex[ip]; uyv += econst*Ey[ip]; uzv += econst*Ez[ip]; }
    ux[ip]=uxv; uy[ip]=uyv; uz[ip]=uzv;
}

void warpx_boris_push_fp64(const double *restrict Bx, const double *restrict By,
                           const double *restrict Bz, const double *restrict Ex,
                           const double *restrict Ey, const double *restrict Ez,
                           double *restrict ux, double *restrict uy,
                           double *restrict uz,
                           const double m, const int64_t momentum_push_type,
                           const int64_t np_particles, const double q,
                           uint8_t *restrict workspace, const int64_t workspace_size)
{
    (void)workspace; (void)workspace_size;
    const double dt = 1.0e-13;   /* fixed config timestep (baked by the reference) */
    const double econst = 0.5 * q * dt / m;
    const int e1 = (momentum_push_type == 1 || momentum_push_type == 0);
    const int e2 = (momentum_push_type == 2 || momentum_push_type == 0);
    const int rescale = (momentum_push_type == 1 || momentum_push_type == 2);
    const long np = (long)np_particles;

#if defined(__AVX512F__)
    const long nvec = np / 8;
    WARPX_OMP
    for (long i = 0; i < nvec; ++i)
        push8(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, i * 8, econst, e1, e2, rescale);
    for (long ip = nvec * 8; ip < np; ++ip)
        push1(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, ip, econst, e1, e2, rescale);
#else
    WARPX_OMP
    for (long ip = 0; ip < np; ++ip)
        push1(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, ip, econst, e1, e2, rescale);
#endif
}
