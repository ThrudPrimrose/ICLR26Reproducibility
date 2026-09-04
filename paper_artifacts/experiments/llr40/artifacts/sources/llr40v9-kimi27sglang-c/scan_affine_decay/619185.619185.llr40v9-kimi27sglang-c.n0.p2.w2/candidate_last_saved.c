#include <stdint.h>
#include <stddef.h>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static inline __m512d vshift_left(__m512d v, int s)
{
    switch (s) {
    case 1:
        return _mm512_permutexvar_pd(_mm512_setr_epi64(7, 0, 1, 2, 3, 4, 5, 6), v);
    case 2:
        return _mm512_permutexvar_pd(_mm512_setr_epi64(6, 7, 0, 1, 2, 3, 4, 5), v);
    case 4:
        return _mm512_permutexvar_pd(_mm512_setr_epi64(4, 5, 6, 7, 0, 1, 2, 3), v);
    default:
        return v;
    }
}

static inline void prefix_scan8(__m512d c, __m512d x, __m512d *cout, __m512d *xout)
{
    __m512d cp = c;
    __m512d xp = x;
    for (int s = 1; s < 8; s <<= 1) {
        __m512d cps = vshift_left(cp, s);
        __m512d xps = vshift_left(xp, s);
        __mmask8 m = (0xFF << s) & 0xFF;
        __m512d cn = _mm512_mul_pd(cp, cps);
        __m512d xn = _mm512_fmadd_pd(cp, xps, xp);
        cp = _mm512_mask_blend_pd(m, cp, cn);
        xp = _mm512_mask_blend_pd(m, xp, xn);
    }
    *cout = cp;
    *xout = xp;
}

static inline double last_lane(__m512d v)
{
    return _mm512_cvtsd_f64(_mm512_permutexvar_pd(_mm512_setr_epi64(7, 7, 7, 7, 7, 7, 7, 7), v));
}

static inline void scan_vec8(__m512d c, __m512d x, double *restrict y, int64_t i, double *carry)
{
    __m512d cp, xp;
    prefix_scan8(c, x, &cp, &xp);
    __m512d yv = _mm512_fmadd_pd(cp, _mm512_set1_pd(*carry), xp);
    if (y) {
        _mm512_storeu_pd(y + i, yv);
    }
    *carry = last_lane(yv);
}

void scan_affine_decay_fp64(const double *restrict c,
                            const double *restrict x,
                            double *restrict y,
                            int64_t n,
                            uint8_t *restrict workspace,
                            int64_t workspace_size)
{
    if (n <= 0) {
        return;
    }

    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
    if (nt < 1) nt = 1;
#endif
    if (nt > n) nt = (int)n;

    const int64_t chunk = (n + nt - 1) / nt;
    const int blocks = (int)((n + chunk - 1) / chunk);

    double *restrict AB;
    double local_ab[64];
    int64_t need = (int64_t)blocks * 2 * (int64_t)sizeof(double);
    if (workspace != (uint8_t *)0 && workspace_size >= need) {
        AB = (double *restrict)workspace;
    } else {
        if ((size_t)need <= sizeof(local_ab)) {
            AB = local_ab;
        } else {
            AB = (double *restrict)__builtin_alloca((size_t)need);
        }
    }

    /* Pass 1: compute affine transform of each chunk. */
    #pragma omp parallel for schedule(static, 1)
    for (int b = 0; b < blocks; ++b) {
        int64_t l = (int64_t)b * chunk;
        int64_t r = l + chunk;
        if (r > n) r = n;

        double carry = 0.0;
        double a = 1.0;
        int64_t i = l;
        for (; i + 8 <= r; i += 8) {
            __m512d cv = _mm512_loadu_pd(c + i);
            __m512d xv = _mm512_loadu_pd(x + i);
            __m512d cp, xp;
            prefix_scan8(cv, xv, &cp, &xp);
            carry = last_lane(_mm512_fmadd_pd(cp, _mm512_set1_pd(carry), xp));
            a *= last_lane(cp);
        }
        for (; i < r; ++i) {
            carry = c[i] * carry + x[i];
            a *= c[i];
        }
        AB[2 * b] = a;
        AB[2 * b + 1] = carry;
    }

    /* Pass 2: serial prefix of chunk transforms. */
    double carry = 0.0;
    for (int b = 0; b < blocks; ++b) {
        double a = AB[2 * b];
        double bval = AB[2 * b + 1];
        AB[2 * b + 1] = carry;
        carry = a * carry + bval;
    }

    /* Pass 3: write final outputs. */
    #pragma omp parallel for schedule(static, 1)
    for (int b = 0; b < blocks; ++b) {
        int64_t l = (int64_t)b * chunk;
        int64_t r = l + chunk;
        if (r > n) r = n;
        double carry = AB[2 * b + 1];
        int64_t i = l;
        for (; i + 8 <= r; i += 8) {
            __m512d cv = _mm512_loadu_pd(c + i);
            __m512d xv = _mm512_loadu_pd(x + i);
            scan_vec8(cv, xv, y, i, &carry);
        }
        for (; i < r; ++i) {
            carry = c[i] * carry + x[i];
            y[i] = carry;
        }
    }
}
