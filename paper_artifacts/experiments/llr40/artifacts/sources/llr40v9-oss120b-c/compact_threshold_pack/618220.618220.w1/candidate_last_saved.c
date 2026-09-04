#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <omp.h>

/* Double-precision version */
void __attribute__((noinline)) compact_threshold_pack_fp64(int64_t * out_count,
                                 double * packed,
                                 const double * src,
                                 const double * weight,
                                 int64_t LEN_1D) {
    /* entry debug removed */
    int64_t n = 0;
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double s = src[i];
        if (s > 0.0) {
            packed[n] = s * weight[i];
            ++n;
        }
    }
    /* result debug removed */
    if (out_count) out_count[0] = n;
}

/* Float-precision version */
void __attribute__((noinline)) compact_threshold_pack_fp32(int64_t * out_count,
                                 float * packed,
                                 const float * src,
                                 const float * weight,
                                 int64_t LEN_1D) {
    /* entry debug removed */
    int64_t n = 0;
    for (int64_t i = 0; i < LEN_1D; ++i) {
        float s = src[i];
        if (s > 0.0f) {
            packed[n] = s * weight[i];
            ++n;
        }
    }
    /* result debug removed */
    if (out_count) out_count[0] = n;
}

/* int64 version */
void __attribute__((noinline)) compact_threshold_pack_i64(int64_t * out_count,
                                 int64_t * packed,
                                 const int64_t * src,
                                 const int64_t * weight,
                                 int64_t LEN_1D) {
    /* entry debug removed */
    int64_t n = 0;
    for (int64_t i = 0; i < LEN_1D; ++i) {
        int64_t s = src[i];
        if (s > 0) {
            packed[n] = s * weight[i];
            ++n;
        }
    }
    /* result debug removed */
    if (out_count) out_count[0] = n;
}

/* int32 version */
void __attribute__((noinline)) compact_threshold_pack_i32(int64_t * out_count,
                                 int32_t * packed,
                                 const int32_t * src,
                                 const int32_t * weight,
                                 int64_t LEN_1D) {
    /* entry debug removed */
    int64_t n = 0;
    for (int64_t i = 0; i < LEN_1D; ++i) {
        int32_t s = src[i];
        if (s > 0) {
            packed[n] = s * weight[i];
            ++n;
        }
    }
    /* result debug removed */
    if (out_count) out_count[0] = n;
}
