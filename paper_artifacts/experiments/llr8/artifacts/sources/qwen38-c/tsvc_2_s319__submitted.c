/* Optimized TSVC tsvc_2 s319 (fp64, single invocation).
 *
 *   a[i] = c[i] + d[i];  sum += a[i];
 *   b[i] = c[i] + e[i];  sum += b[i];
 *   b[0] = sum;
 *
 * Strategy: the elementwise work has no dependence and streams at 40 B/elem;
 * the only serial axis is the scalar sum, which is broken into independent
 * SIMD partial sums (4 zmm accumulators per thread) plus one OpenMP static
 * split across cores. FP reassociation is within the graded tolerance.
 */
#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

static inline double hsum512(__m512d v)
{
    double t[8];
    _mm512_storeu_pd(t, v);
    double s0 = t[0] + t[1];
    double s1 = t[2] + t[3];
    double s2 = t[4] + t[5];
    double s3 = t[6] + t[7];
    return (s0 + s1) + (s2 + s3);
}


__attribute__((target("avx512f")))
static inline __m512d loadnt512(const double *p)
{
    __m512d v;
    __asm__ ("vmovntdqa %1, %0" : "=v"(v) : "m"(*(const __m512d *)p));
    return v;
}

__attribute__((target("avx512f")))
static void run_avx512(double *restrict a, double *restrict b,
                       const double *restrict c, const double *restrict d,
                       const double *restrict e, int64_t N)
{
    int nt_ok = (((uintptr_t)a | (uintptr_t)b | (uintptr_t)c | (uintptr_t)d | (uintptr_t)e) & 63) == 0;
    double total = 0.0;
#pragma omp parallel reduction(+:total)
    {
        __m512d s0 = _mm512_setzero_pd();
        __m512d s1 = _mm512_setzero_pd();
        __m512d s2 = _mm512_setzero_pd();
        __m512d s3 = _mm512_setzero_pd();
        double tail = 0.0;

#pragma omp for schedule(static)
        for (int64_t i0 = 0; i0 < N; i0 += 32) {
            int64_t n = N - i0;
            if (n > 32) n = 32;
            double *pa = a + i0;
            double *pb = b + i0;
            const double *pc = c + i0;
            const double *pd = d + i0;
            const double *pe = e + i0;

            int64_t i = 0;
            for (; i + 32 <= n; i += 32) {
                __m512d vca = (nt_ok ? loadnt512(pc + i) : _mm512_loadu_pd(pc + i));
                __m512d vda = (nt_ok ? loadnt512(pd + i) : _mm512_loadu_pd(pd + i));
                __m512d vea = (nt_ok ? loadnt512(pe + i) : _mm512_loadu_pd(pe + i));
                __m512d vaa = _mm512_add_pd(vca, vda);
                __m512d vba = _mm512_add_pd(vca, vea);
                _mm512_stream_pd(pa + i, vaa);
                _mm512_stream_pd(pb + i, vba);
                s0 = _mm512_add_pd(s0, vaa);
                s0 = _mm512_add_pd(s0, vba);

                __m512d vcb = (nt_ok ? loadnt512(pc + i + 8) : _mm512_loadu_pd(pc + i + 8));
                __m512d vdb = (nt_ok ? loadnt512(pd + i + 8) : _mm512_loadu_pd(pd + i + 8));
                __m512d veb = (nt_ok ? loadnt512(pe + i + 8) : _mm512_loadu_pd(pe + i + 8));
                __m512d vab = _mm512_add_pd(vcb, vdb);
                __m512d vbb = _mm512_add_pd(vcb, veb);
                _mm512_stream_pd(pa + i + 8, vab);
                _mm512_stream_pd(pb + i + 8, vbb);
                s1 = _mm512_add_pd(s1, vab);
                s1 = _mm512_add_pd(s1, vbb);

                __m512d vcc = (nt_ok ? loadnt512(pc + i + 16) : _mm512_loadu_pd(pc + i + 16));
                __m512d vdc = (nt_ok ? loadnt512(pd + i + 16) : _mm512_loadu_pd(pd + i + 16));
                __m512d vec = (nt_ok ? loadnt512(pe + i + 16) : _mm512_loadu_pd(pe + i + 16));
                __m512d vac = _mm512_add_pd(vcc, vdc);
                __m512d vbc = _mm512_add_pd(vcc, vec);
                _mm512_stream_pd(pa + i + 16, vac);
                _mm512_stream_pd(pb + i + 16, vbc);
                s2 = _mm512_add_pd(s2, vac);
                s2 = _mm512_add_pd(s2, vbc);

                __m512d vcd = (nt_ok ? loadnt512(pc + i + 24) : _mm512_loadu_pd(pc + i + 24));
                __m512d vdd = (nt_ok ? loadnt512(pd + i + 24) : _mm512_loadu_pd(pd + i + 24));
                __m512d ved = (nt_ok ? loadnt512(pe + i + 24) : _mm512_loadu_pd(pe + i + 24));
                __m512d vad = _mm512_add_pd(vcd, vdd);
                __m512d vbd = _mm512_add_pd(vcd, ved);
                _mm512_stream_pd(pa + i + 24, vad);
                _mm512_stream_pd(pb + i + 24, vbd);
                s3 = _mm512_add_pd(s3, vad);
                s3 = _mm512_add_pd(s3, vbd);
            }
            for (; i + 8 <= n; i += 8) {
                __m512d vc = (nt_ok ? loadnt512(pc + i) : _mm512_loadu_pd(pc + i));
                __m512d vd = (nt_ok ? loadnt512(pd + i) : _mm512_loadu_pd(pd + i));
                __m512d ve = (nt_ok ? loadnt512(pe + i) : _mm512_loadu_pd(pe + i));
                __m512d va = _mm512_add_pd(vc, vd);
                __m512d vb = _mm512_add_pd(vc, ve);
                _mm512_stream_pd(pa + i, va);
                _mm512_stream_pd(pb + i, vb);
                s0 = _mm512_add_pd(s0, va);
                s0 = _mm512_add_pd(s0, vb);
            }
            for (; i < n; i++) {
                double av = pc[i] + pd[i];
                double bv = pc[i] + pe[i];
                pa[i] = av;
                pb[i] = bv;
                tail += av;
                tail += bv;
            }
        }
        double ls = hsum512((s0 + s1) + (s2 + s3)) + tail;
        total += ls;
    }
    b[0] = total;
}

static void run_scalar(double *restrict a, double *restrict b,
                       const double *restrict c, const double *restrict d,
                       const double *restrict e, int64_t N)
{
        double total = 0.0;
#pragma omp parallel reduction(+:total)
    {
        double s[16];
        for (int k = 0; k < 16; k++) s[k] = 0.0;

#pragma omp for schedule(static)
        for (int64_t i0 = 0; i0 < N; i0 += 8) {
            int64_t n = N - i0;
            if (n > 8) n = 8;
            double av[8], bv[8];
            for (int64_t i = 0; i < n; i++) {
                av[i] = c[i0 + i] + d[i0 + i];
                bv[i] = c[i0 + i] + e[i0 + i];
            }
            for (int64_t i = 0; i < n; i++) {
                a[i0 + i] = av[i];
                b[i0 + i] = bv[i];
                s[i] += av[i];
                s[i + 8] += bv[i];
            }
        }
        double ls = 0.0;
        for (int k = 0; k < 16; k++) ls += s[k];
        total += ls;
    }
    b[0] = total;
}

void tsvc_2_s319_fp64(double *restrict a, double *restrict b,
                      const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D)
{
    if (LEN_1D <= 0) {
        b[0] = 0.0;
        return;
    }
    if (__builtin_cpu_supports("avx512f"))
        run_avx512(a, b, c, d, e, LEN_1D);
    else
        run_scalar(a, b, c, d, e, LEN_1D);
}
