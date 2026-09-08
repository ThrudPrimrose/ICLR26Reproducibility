#include <stdint.h>
#include <omp.h>
#include <immintrin.h>
#include <stdint.h>

void tsvc_2_s1232_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D,
                       const int64_t VLEN) {

    #pragma omp parallel
    {
        #pragma omp for schedule(guided) nowait
        for (int64_t i = 0; i < LEN_2D; ++i) {
            const int64_t jmax = i / VLEN;
            double *restrict a_row = aa + i * LEN_2D;
            const double *restrict b_row = bb + i * LEN_2D;
            const double *restrict c_row = cc + i * LEN_2D;

            int64_t j = 0;
            while (j <= jmax && (((uintptr_t)&a_row[j]) & 0x1f) != 0) {
                a_row[j] = b_row[j] + c_row[j];
                ++j;
            }

            const int64_t vec_end = jmax - 3;
            for (; j <= vec_end; j += 4) {
                __m256d vb = _mm256_loadu_pd(&b_row[j]);
                __m256d vc = _mm256_loadu_pd(&c_row[j]);
                __m256d va = _mm256_add_pd(vb, vc);
                _mm256_stream_pd(&a_row[j], va);
            }

            for (; j <= jmax; ++j) {
                a_row[j] = b_row[j] + c_row[j];
            }
        }
        _mm_sfence();
        #pragma omp barrier
    }
}
