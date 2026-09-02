#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void quasi_affine_reduce_odd_fp64(double *restrict a,
                                  double *restrict out,
                                  int64_t LEN_1D,
                                  uint8_t *restrict workspace,
                                  int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;

    int nthreads = omp_get_num_procs();
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 256) nthreads = 256;

    double partial[256];

    int64_t n_odd = LEN_1D / 2;
    if (n_odd <= 0) {
        *out = 0.0;
        return;
    }

    const __mmask8 odd_mask = 0xAA; // lanes 1,3,5,7 are odd global indices

    #pragma omp parallel num_threads(nthreads) default(none) shared(a, n_odd, partial, odd_mask)
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        int64_t chunk = n_odd / nt;
        int64_t rem = n_odd % nt;
        int64_t start, end;
        if (tid < rem) {
            start = tid * (chunk + 1);
            end = start + (chunk + 1);
        } else {
            start = rem * (chunk + 1) + (tid - rem) * chunk;
            end = start + chunk;
        }

        int64_t j0 = 2 * start;         // even base for vector loads
        int64_t j1 = 2 * end;           // one past the last even base (exclusive)

        __m512d acc = _mm512_setzero_pd();
        int64_t j;
        for (j = j0; j + 8 <= j1; j += 8) {
            __m512d v = _mm512_loadu_pd(a + j);
            acc = _mm512_mask_add_pd(acc, odd_mask, acc, v);
        }
        double s = _mm512_reduce_add_pd(acc);

        // scalar tail: first odd index not covered by the vector loop
        int64_t first_odd = j + 1;
        int64_t last_odd = j1 - 1;
        for (int64_t idx = first_odd; idx <= last_odd; idx += 2) {
            s += a[idx];
        }

        partial[tid] = s;
    }

    double sum = 0.0;
    for (int t = 0; t < nthreads; ++t) {
        sum += partial[t];
    }
    *out = sum;
}
