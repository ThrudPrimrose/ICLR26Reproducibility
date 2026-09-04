#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_s316_fp64(const double *restrict a, double *restrict result, const int64_t LEN_1D) {
    double global_min = a[0];
    double local_mins[128];

    if (LEN_1D > 1) {
        #pragma omp parallel
        {
            int nt = omp_get_num_threads();
            int tid = omp_get_thread_num();
            int64_t n = LEN_1D - 1;
            int64_t chunk = n / nt;
            int64_t rem = n % nt;
            int64_t start = 1 + tid * chunk + (tid < rem ? tid : rem);
            int64_t end = start + chunk + (tid < rem ? 1 : 0);

            __m512d vx = _mm512_set1_pd(global_min);
            int64_t i = start;

            for (; i + 8 <= end; i += 8) {
                __m512d va = _mm512_loadu_pd(&a[i]);
                __mmask8 lt = _mm512_cmp_pd_mask(va, vx, _CMP_LT_OQ);
                vx = _mm512_mask_blend_pd(lt, vx, va);
            }

            double lanes[8] __attribute__((aligned(64)));
            _mm512_store_pd(lanes, vx);
            double local_min = lanes[0];
            for (int j = 1; j < 8; ++j) {
                if (lanes[j] < local_min) {
                    local_min = lanes[j];
                }
            }

            for (; i < end; ++i) {
                if (a[i] < local_min) {
                    local_min = a[i];
                }
            }

            local_mins[tid] = local_min;
            #pragma omp barrier
            #pragma omp single
            {
                for (int t = 0; t < nt; ++t) {
                    if (local_mins[t] < global_min) {
                        global_min = local_mins[t];
                    }
                }
            }
        }
    }

    result[0] = global_min;
}
