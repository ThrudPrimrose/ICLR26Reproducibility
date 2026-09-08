#include <cstdint>

// Forward declaration of the device implementation (defined in device_source)
extern "C" void tsvc_2_s311_fp64_device(const double * __restrict__ a,
                                         double * __restrict__ sum_out,
                                         const int64_t LEN_1D);

// Exported entry point required by the benchmark harness.
extern "C" void tsvc_2_s311_fp64(const double * __restrict__ a,
                                   double * __restrict__ sum_out,
                                   const int64_t LEN_1D) {
    // Delegate to the device-side implementation.
    tsvc_2_s311_fp64_device(a, sum_out, LEN_1D);
}
