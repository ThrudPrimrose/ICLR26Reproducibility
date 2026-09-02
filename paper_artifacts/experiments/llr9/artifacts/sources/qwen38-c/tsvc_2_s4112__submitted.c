/* TSVC tsvc_2 s4112: a[i] = a[i] + b[ip[i]] * 2.0   (ip: int32 indices)
 *
 * Random gather over b (~612MB, exceeds L3): latency-bound. Hide the ~400cy
 * DRAM latency with (a) one OpenMP thread per core (each contributes
 * independent in-flight misses) and (b) two-level software prefetch (lead
 * 512 and 1024 elements, 8 wide) ahead of the consuming loads. 8-wide
 * unroll keeps the a/ip streams sequential.
 */
#include <stdint.h>
#include <omp.h>

#ifndef NT
#define NT 24
#endif

void tsvc_2_s4112_fp64(double *restrict a, const double *restrict b,
                       const int32_t *restrict ip, const int64_t LEN_1D,
                       unsigned char *restrict workspace, const int64_t workspace_size)
{
    (void)workspace; (void)workspace_size;
    const int64_t N = LEN_1D;
    if (N <= 0) return;

    omp_set_num_threads(NT);

    const int64_t D1 = 512, D2 = 1024;
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nt  = omp_get_num_threads();
        int64_t beg = (N * (int64_t)tid) / nt;
        int64_t end = (N * (int64_t)(tid + 1)) / nt;
        const int64_t pf_end = (N - D2 - 8) > 0 ? (N - D2 - 8) : 0;
        int64_t i = beg;
        const int64_t i_end8 = beg + ((end - beg) & ~(int64_t)7);
        for (; i < i_end8; i += 8) {
            if (i < pf_end) {
                #pragma GCC unroll 8
                for (int k = 0; k < 8; k++)
                    __builtin_prefetch(&b[ip[i + D1 + k]], 0, 1);
                #pragma GCC unroll 8
                for (int k = 0; k < 8; k++)
                    __builtin_prefetch(&b[ip[i + D2 + k]], 0, 1);
            }
            #pragma GCC unroll 8
            for (int k = 0; k < 8; k++)
                a[i + k] += 2.0 * b[ip[i + k]];
        }
        for (; i < end; i++) a[i] += 2.0 * b[ip[i]];
    }
}
