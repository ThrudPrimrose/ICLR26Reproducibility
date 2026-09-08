#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

/*
 * Stream compaction kernel: pack src[i] * weight[i] for each src[i] > 0.
 * The output count is written to out_count[0].
 *
 * Two variants are provided for double (fp64) and float (fp32) data types.
 * The function signatures follow the convention used by other kernels in the suite:
 *   <benchmark>_<kernel>_fp64   and   <benchmark>_<kernel>_fp32
 * where the benchmark prefix for this task is omitted because the kernel key
 * already includes the benchmark name (loop_level_reasoning).
 */

static void compact_threshold_pack_generic(const double *restrict src,
                                            const double *restrict weight,
                                            double *restrict packed,
                                            int64_t *out_count,
                                            int64_t LEN_1D)
{
    if (LEN_1D <= 0) {
        if (out_count) *out_count = 0;
        return;
    }
    int64_t n = 0;
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (src[i] > 0.0) {
            packed[n++] = src[i] * weight[i];
        }
    }
    // Zero remaining entries
    for (int64_t i = n; i < LEN_1D; ++i) {
        packed[i] = 0.0;
    }
    if (out_count) *out_count = n;
    return;
}

/* Double‑precision entry point required by the benchmark harness. */
void compact_threshold_pack_fp64(int64_t *out_index,
                                const double *restrict src,
                                const double *restrict weight,
                                double *restrict packed,
                                int64_t LEN_1D,
                                const uint8_t *restrict mask,
                                int64_t *out_count)
{
    (void)out_index; // Unused
    (void)mask; // Unused
    compact_threshold_pack_generic(src, weight, packed, out_count, LEN_1D);
}

/* Single‑precision variant – identical logic but with float arithmetic. */
static void compact_threshold_pack_generic_f32(const float *restrict src,
                                               const float *restrict weight,
                                               float *restrict packed,
                                               int64_t *out_count,
                                               int64_t LEN_1D)
{
    if (LEN_1D <= 0) {
        if (out_count) *out_count = 0;
        return;
    }
    int64_t n = 0;
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (src[i] > 0.0f) {
            packed[n++] = src[i] * weight[i];
        }
    }
    // Zero remaining entries
    for (int64_t i = n; i < LEN_1D; ++i) {
        packed[i] = 0.0f;
    }
    if (out_count) *out_count = n;
    return;
}

void compact_threshold_pack_fp32(int64_t *out_index,
                                const float *restrict src,
                                const float *restrict weight,
                                float *restrict packed,
                                int64_t LEN_1D,
                                const uint8_t *restrict mask,
                                int64_t *out_count)
{
    (void)out_index; // Unused
    (void)mask; // Unused
    compact_threshold_pack_generic_f32(src, weight, packed, out_count, LEN_1D);
}

