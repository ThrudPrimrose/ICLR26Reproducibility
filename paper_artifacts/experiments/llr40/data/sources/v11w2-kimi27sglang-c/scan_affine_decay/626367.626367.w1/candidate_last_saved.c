#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>

static inline __m512d shift_in_ones(__m512d v, __m512i idx) {
    const __m512d ones = _mm512_set1_pd(1.0);
    return _mm512_permutex2var_pd(v, idx, ones);
}

static inline __m512d shift_in_zeros(__m512d v, __m512i idx) {
    const __m512d zeros = _mm512_setzero_pd();
    return _mm512_permutex2var_pd(v, idx, zeros);
}

void scan_affine_decay_fp64(const double *restrict c, const double *restrict x, double *restrict y, const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;
    y[0] = x[0];
    if (LEN_1D <= 1) return;

    const int64_t n = LEN_1D;

    if (n < 4096) {
        for (int64_t i = 1; i < n; ++i) {
            y[i] = c[i] * y[i - 1] + x[i];
        }
        return;
    }

    const int64_t block_size = 4096;
    const int64_t nblocks = (n + block_size - 1) / block_size;

    double *restrict ablock = (double *)malloc(sizeof(double) * nblocks);
    double *restrict bblock = (double *)malloc(sizeof(double) * nblocks);
    double *restrict carry_in = (double *)malloc(sizeof(double) * nblocks);

    const __m512i idx1 = _mm512_set_epi64(6, 5, 4, 3, 2, 1, 0, 8);
    const __m512i idx2 = _mm512_set_epi64(5, 4, 3, 2, 1, 0, 9, 8);
    const __m512i idx4 = _mm512_set_epi64(3, 2, 1, 0, 11, 10, 9, 8);

    // First pass: local scan per block (carry=0) and block totals.
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < nblocks; ++b) {
        int64_t s = b * block_size;
        int64_t e = s + block_size;
        if (e > n) e = n;

        double acc = 0.0;
        int64_t i = s;
        for (; i + 8 <= e; i += 8) {
            __m512d cv = _mm512_loadu_pd(&c[i]);
            __m512d bv = _mm512_loadu_pd(&x[i]);
            __m512d av = cv;

            // prefix combine over 8 lanes, identity (1,0) before lane 0
            __m512d as = shift_in_ones(av, idx1);
            __m512d bs = shift_in_zeros(bv, idx1);
            bv = _mm512_fmadd_pd(av, bs, bv);
            av = _mm512_mul_pd(av, as);

            as = shift_in_ones(av, idx2);
            bs = shift_in_zeros(bv, idx2);
            bv = _mm512_fmadd_pd(av, bs, bv);
            av = _mm512_mul_pd(av, as);

            as = shift_in_ones(av, idx4);
            bs = shift_in_zeros(bv, idx4);
            bv = _mm512_fmadd_pd(av, bs, bv);
            av = _mm512_mul_pd(av, as);

            __m512d yv = _mm512_fmadd_pd(av, _mm512_set1_pd(acc), bv);
            _mm512_storeu_pd(&y[i], yv);

            double tmp[8];
            _mm512_storeu_pd(tmp, yv);
            acc = tmp[7];
        }
        for (; i < e; ++i) {
            acc = c[i] * acc + x[i];
            y[i] = acc;
        }

        // a_block: product of c[s..e-1]
        double a_prod = 1.0;
        for (int64_t j = s; j < e; ++j) a_prod *= c[j];
        ablock[b] = a_prod;
        bblock[b] = acc;
    }

    // Exclusive prefix over block totals.
    double carry = 0.0;
    for (int64_t b = 0; b < nblocks; ++b) {
        carry_in[b] = carry;
        carry = ablock[b] * carry + bblock[b];
    }

    // Second pass: apply incoming carry using prefix products of c.
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < nblocks; ++b) {
        int64_t s = b * block_size;
        int64_t e = s + block_size;
        if (e > n) e = n;

        double carry = carry_in[b];
        __m512d carryv = _mm512_set1_pd(carry);
        double a_run = 1.0;
        int64_t i = s;
        for (; i + 8 <= e; i += 8) {
            __m512d cv = _mm512_loadu_pd(&c[i]);
            __m512d av = cv;

            av = _mm512_mul_pd(av, shift_in_ones(av, idx1));
            av = _mm512_mul_pd(av, shift_in_ones(av, idx2));
            av = _mm512_mul_pd(av, shift_in_ones(av, idx4));

            __m512d a_total = _mm512_mul_pd(av, _mm512_set1_pd(a_run));
            __m512d yv = _mm512_loadu_pd(&y[i]);
            yv = _mm512_fmadd_pd(a_total, carryv, yv);
            _mm512_storeu_pd(&y[i], yv);

            double tmp[8];
            _mm512_storeu_pd(tmp, av);
            a_run *= tmp[7];
        }
        for (; i < e; ++i) {
            a_run *= c[i];
            y[i] = a_run * carry + y[i];
        }
    }

    free(ablock);
    free(bblock);
    free(carry_in);
}
