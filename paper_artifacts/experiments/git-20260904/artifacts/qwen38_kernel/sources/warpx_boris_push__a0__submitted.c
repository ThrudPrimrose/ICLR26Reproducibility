#include <math.h>
#include <stdint.h>
#include <immintrin.h>

#define C_LIGHT 299792458.0
#define INV_C2 (1.0 / (C_LIGHT * C_LIGHT))
#define DT_PINS 1.0e-13   /* dt pinned to 1.0e-13 by the manifest (not passed) */

/* Boris rotation core: given current u (ux,uy,uz) and t (tx,ty,tz) [already
 * rescaled or not], update u in place. Matches the reference math per particle. */
static inline void rot_core(__m512d *ux, __m512d *uy, __m512d *uz,
                            __m512d tx, __m512d ty, __m512d tz) {
    __m512d t2 = _mm512_add_pd(_mm512_fmadd_pd(tz, tz, _mm512_fmadd_pd(ty, ty, _mm512_mul_pd(tx, tx))),
                               _mm512_setzero_pd());
    __m512d tsqi = _mm512_div_pd(_mm512_set1_pd(2.0),
                                 _mm512_add_pd(_mm512_set1_pd(1.0), t2));
    __m512d sx = _mm512_mul_pd(tx, tsqi);
    __m512d sy = _mm512_mul_pd(ty, tsqi);
    __m512d sz = _mm512_mul_pd(tz, tsqi);
    __m512d ux_p = _mm512_sub_pd(_mm512_fmadd_pd(tz, *uy, *ux), _mm512_mul_pd(*uz, ty));
    __m512d uy_p = _mm512_sub_pd(_mm512_fmadd_pd(tx, *uz, *uy), _mm512_mul_pd(*ux, tz));
    __m512d uz_p = _mm512_sub_pd(_mm512_fmadd_pd(ty, *ux, *uz), _mm512_mul_pd(*uy, tx));
    *ux = _mm512_sub_pd(_mm512_fmadd_pd(sz, uy_p, *ux), _mm512_mul_pd(uz_p, sy));
    *uy = _mm512_sub_pd(_mm512_fmadd_pd(sx, uz_p, *uy), _mm512_mul_pd(ux_p, sz));
    *uz = _mm512_sub_pd(_mm512_fmadd_pd(sy, ux_p, *uz), _mm512_mul_pd(uy_p, sx));
}

/* Half-push t rescale: factor = (sqrt(1+|t|^2)-1)/|t|^2 (0.5 where |t|^2==0). */
static inline void rescale_t(__m512d *tx, __m512d *ty, __m512d *tz) {
    __m512d tsq = _mm512_fmadd_pd(*tz, *tz, _mm512_fmadd_pd(*ty, *ty, _mm512_mul_pd(*tx, *tx)));
    __m512d s = _mm512_sqrt_pd(_mm512_add_pd(_mm512_set1_pd(1.0), tsq));
    __m512d f = _mm512_div_pd(_mm512_sub_pd(s, _mm512_set1_pd(1.0)), tsq);
    __mmask8 m = _mm512_cmp_pd_mask(tsq, _mm512_setzero_pd(), _CMP_GT_OQ);
    f = _mm512_mask_blend_pd(m, _mm512_set1_pd(0.5), f);  /* m ? f : 0.5 */
    *tx = _mm512_mul_pd(*tx, f);
    *ty = _mm512_mul_pd(*ty, f);
    *tz = _mm512_mul_pd(*tz, f);
}

static inline void body_full(int64_t ip,
                             const double *__restrict__ Bx, const double *__restrict__ By,
                             const double *__restrict__ Bz, const double *__restrict__ Ex,
                             const double *__restrict__ Ey, const double *__restrict__ Ez,
                             double *__restrict__ ux, double *__restrict__ uy, double *__restrict__ uz,
                             __m512d ec) {
    __m512d Bxv = _mm512_loadu_pd(Bx + ip), Byv = _mm512_loadu_pd(By + ip), Bzv = _mm512_loadu_pd(Bz + ip);
    __m512d Exv = _mm512_loadu_pd(Ex + ip), Eyv = _mm512_loadu_pd(Ey + ip), Ezv = _mm512_loadu_pd(Ez + ip);
    __m512d uxv = _mm512_loadu_pd(ux + ip), uyv = _mm512_loadu_pd(uy + ip), uzv = _mm512_loadu_pd(uz + ip);
    uxv = _mm512_fmadd_pd(ec, Exv, uxv);
    uyv = _mm512_fmadd_pd(ec, Eyv, uyv);
    uzv = _mm512_fmadd_pd(ec, Ezv, uzv);
    __m512d u2 = _mm512_fmadd_pd(uxv, uxv, _mm512_fmadd_pd(uyv, uyv, _mm512_mul_pd(uzv, uzv)));
    __m512d inv_gamma = _mm512_div_pd(_mm512_set1_pd(1.0),
                                      _mm512_sqrt_pd(_mm512_fmadd_pd(_mm512_set1_pd(INV_C2), u2, _mm512_set1_pd(1.0))));
    __m512d ecig = _mm512_mul_pd(ec, inv_gamma);
    __m512d tx = _mm512_mul_pd(ecig, Bxv), ty = _mm512_mul_pd(ecig, Byv), tz = _mm512_mul_pd(ecig, Bzv);
    rot_core(&uxv, &uyv, &uzv, tx, ty, tz);
    uxv = _mm512_fmadd_pd(ec, Exv, uxv);
    uyv = _mm512_fmadd_pd(ec, Eyv, uyv);
    uzv = _mm512_fmadd_pd(ec, Ezv, uzv);
    _mm512_storeu_pd(ux + ip, uxv);
    _mm512_storeu_pd(uy + ip, uyv);
    _mm512_storeu_pd(uz + ip, uzv);
}

static inline void body_half(int64_t ip, int first_half,
                             const double *__restrict__ Bx, const double *__restrict__ By,
                             const double *__restrict__ Bz, const double *__restrict__ Ex,
                             const double *__restrict__ Ey, const double *__restrict__ Ez,
                             double *__restrict__ ux, double *__restrict__ uy, double *__restrict__ uz,
                             __m512d ec) {
    __m512d Bxv = _mm512_loadu_pd(Bx + ip), Byv = _mm512_loadu_pd(By + ip), Bzv = _mm512_loadu_pd(Bz + ip);
    __m512d Exv = _mm512_loadu_pd(Ex + ip), Eyv = _mm512_loadu_pd(Ey + ip), Ezv = _mm512_loadu_pd(Ez + ip);
    __m512d uxv = _mm512_loadu_pd(ux + ip), uyv = _mm512_loadu_pd(uy + ip), uzv = _mm512_loadu_pd(uz + ip);
    if (first_half) {
        uxv = _mm512_fmadd_pd(ec, Exv, uxv);
        uyv = _mm512_fmadd_pd(ec, Eyv, uyv);
        uzv = _mm512_fmadd_pd(ec, Ezv, uzv);
    }
    __m512d u2 = _mm512_fmadd_pd(uxv, uxv, _mm512_fmadd_pd(uyv, uyv, _mm512_mul_pd(uzv, uzv)));
    __m512d inv_gamma = _mm512_div_pd(_mm512_set1_pd(1.0),
                                      _mm512_sqrt_pd(_mm512_fmadd_pd(_mm512_set1_pd(INV_C2), u2, _mm512_set1_pd(1.0))));
    __m512d ecig = _mm512_mul_pd(ec, inv_gamma);
    __m512d tx = _mm512_mul_pd(ecig, Bxv), ty = _mm512_mul_pd(ecig, Byv), tz = _mm512_mul_pd(ecig, Bzv);
    rescale_t(&tx, &ty, &tz);
    rot_core(&uxv, &uyv, &uzv, tx, ty, tz);
    if (!first_half) {
        uxv = _mm512_fmadd_pd(ec, Exv, uxv);
        uyv = _mm512_fmadd_pd(ec, Eyv, uyv);
        uzv = _mm512_fmadd_pd(ec, Ezv, uzv);
    }
    _mm512_storeu_pd(ux + ip, uxv);
    _mm512_storeu_pd(uy + ip, uyv);
    _mm512_storeu_pd(uz + ip, uzv);
}

/* Scalar fallback for the tail (<8) particles: exact reference math. */
static void scalar_loop(const double *__restrict__ Bx, const double *__restrict__ By,
                        const double *__restrict__ Bz, const double *__restrict__ Ex,
                        const double *__restrict__ Ey, const double *__restrict__ Ez,
                        double *__restrict__ ux, double *__restrict__ uy, double *__restrict__ uz,
                        double dt, double m, int momentum_push_type, double q, long start, long np) {
    const double econst = 0.5 * q * dt / m;
    for (long ip = start; ip < start + np; ++ip) {
        double a = ux[ip], b = uy[ip], c = uz[ip];
        if (momentum_push_type == 1 || momentum_push_type == 0) {
            a += econst * Ex[ip]; b += econst * Ey[ip]; c += econst * Ez[ip];
        }
        double inv_gamma = 1.0 / sqrt(1.0 + (a * a + b * b + c * c) * INV_C2);
        double tx = econst * inv_gamma * Bx[ip], ty = econst * inv_gamma * By[ip], tz = econst * inv_gamma * Bz[ip];
        if (momentum_push_type == 1 || momentum_push_type == 2) {
            double tsq = tx * tx + ty * ty + tz * tz;
            double factor = (tsq > 0.0) ? (sqrt(1.0 + tsq) - 1.0) / tsq : 0.5;
            tx *= factor; ty *= factor; tz *= factor;
        }
        double tsqi = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz);
        double sx = tx * tsqi, sy = ty * tsqi, sz = tz * tsqi;
        double ux_p = a + b * tz - c * ty;
        double uy_p = b + c * tx - a * tz;
        double uz_p = c + a * ty - b * tx;
        a += uy_p * sz - uz_p * sy;
        b += uz_p * sx - ux_p * sz;
        c += ux_p * sy - uy_p * sx;
        if (momentum_push_type == 2 || momentum_push_type == 0) {
            a += econst * Ex[ip]; b += econst * Ey[ip]; c += econst * Ez[ip];
        }
        ux[ip] = a; uy[ip] = b; uz[ip] = c;
    }
}

void warpx_boris_push_fp64(const double *__restrict__ Bx, const double *__restrict__ By,
                           const double *__restrict__ Bz, const double *__restrict__ Ex,
                           const double *__restrict__ Ey, const double *__restrict__ Ez,
                           double *__restrict__ ux, double *__restrict__ uy, double *__restrict__ uz,
                           double m, int64_t momentum_push_type, int64_t np_particles, double q,
                           uint8_t *workspace, int64_t workspace_size) {
    (void)workspace; (void)workspace_size;
    const long np = (long)np_particles;
    if (np <= 0) return;
    const int mpt = (int)momentum_push_type;
    const double econst = 0.5 * q * DT_PINS / m;
    const __m512d ec = _mm512_set1_pd(econst);
    const long nfull = np >> 3;
    const long rem = np & 7;

    if (mpt == 0) {
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < nfull; ++i)
            body_full(i * 8, Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, ec);
        if (rem) scalar_loop(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, DT_PINS, m, mpt, q, nfull * 8, rem);
    } else if (mpt == 1) {
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < nfull; ++i)
            body_half(i * 8, 1, Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, ec);
        if (rem) scalar_loop(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, DT_PINS, m, mpt, q, nfull * 8, rem);
    } else {
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < nfull; ++i)
            body_half(i * 8, 0, Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, ec);
        if (rem) scalar_loop(Bx, By, Bz, Ex, Ey, Ez, ux, uy, uz, DT_PINS, m, mpt, q, nfull * 8, rem);
    }
}
