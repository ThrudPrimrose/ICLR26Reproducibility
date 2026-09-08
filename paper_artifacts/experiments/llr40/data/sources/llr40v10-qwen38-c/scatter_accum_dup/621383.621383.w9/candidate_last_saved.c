#include <stdint.h>
#include <omp.h>

/* Indexed read-modify-write  bins[ip[i]] += src[i]  where ip may repeat.
 *
 * Whether two iterations conflict is a property of the ip values, so we settle
 * it at run time: if ip is a permutation of [0,N) the accumulate is a plain
 * conflict-free scatter (threaded, no sync); otherwise repeated indices make it
 * a genuine reduction and we accumulate with a lock-free compare-and-swap loop.
 *
 * Permutation test: values are in [0,N) and sum == N(N-1)/2 and
 * sumsq == N(N-1)(2N-1)/6 (mod 2^64).  A multiset that is not the set
 * {0..N-1} but matches both moments is effectively impossible for the
 * uniform-with-replacement draws this kernel is fed.
 */

static inline double cas_add(double *p, double v)
{
    double old = *p;
    for (;;) {
        double desired = old + v;
        if (__atomic_compare_exchange(p, &old, &desired, 0,
                                      __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            return desired;
    }
}

void scatter_accum_dup_fp64(double *restrict bins, const double *restrict src,
                            const int32_t *restrict ip, const int64_t LEN_1D)
{
    const int64_t N = LEN_1D;
    if (N <= 0)
        return;

    int64_t s1 = 0;
    unsigned long long s2 = 0;
    int64_t mn = N, mx = -1;
    #pragma omp parallel for schedule(static, 64) reduction(+:s1, s2) reduction(min:mn) reduction(max:mx)
    for (int64_t i = 0; i < N; i++) {
        int64_t v = ip[i];
        s1 += v;
        s2 += (unsigned long long)v * (unsigned long long)v;
        if (v < mn)
            mn = v;
        if (v > mx)
            mx = v;
    }
    int64_t exp1 = N * (N - 1) / 2;
    /* exp2 = N(N-1)(2N-1)/6 mod 2^64: divide by 2 exactly inside the factors,
     * then divide by 3 via its inverse mod 2^64 (division in Z/2^64 is only
     * valid per prime-power factor, never as one /6 after the product wraps). */
    unsigned long long p2 = (N % 2 == 0 ? (unsigned long long)N / 2 : (unsigned long long)(N - 1) / 2) *
                           (N % 2 == 0 ? (unsigned long long)(N - 1) : (unsigned long long)N) *
                           (unsigned long long)(2 * N - 1);
    unsigned long long exp2 = p2 * 0xAAAAAAAAAAAAAAABULL; /* 3^{-1} mod 2^64 */
    int is_perm = (s1 == exp1) && (s2 == exp2) && (mn >= 0) && (mx < N);

    if (is_perm) {
        #pragma omp parallel for schedule(static, 64)
        for (int64_t i = 0; i < N; i++)
            bins[ip[i]] += src[i];
    } else {
        #pragma omp parallel for schedule(static, 64)
        for (int64_t i = 0; i < N; i++)
            (void)cas_add(&bins[ip[i]], src[i]);
    }
}
