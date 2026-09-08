// Host half: entry point matching the judge's C ABI, does nothing but launch.
#include <cstdint>

// Launcher lives in the device half (scatter_accum_dup.hip); declared here so the units link.
extern "C" void scatter_accum_dup_launch(double* bins, const int32_t* ip, const double* src, const int64_t n);

extern "C" void scatter_accum_dup_fp64(double* __restrict__ bins,
                                       const int32_t* __restrict__ ip,
                                       const double* __restrict__ src,
                                       const int64_t LEN_1D) {
    scatter_accum_dup_launch(bins, ip, src, LEN_1D);
}
