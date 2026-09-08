#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <cstdint>
#include <limits>
#include <cstdio>

// Simple error checking macro
#define CHECKHIP(call)                                   \
    do {                                                \
        hipError_t err = (call);                        \
        if (err != hipSuccess) {                       \
            fprintf(stderr, "HIP error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__);\
            return;                                    \
        }                                               \
    } while (0)

extern "C" void tsvc_2_s316_fp64(const double *a, double *result, const int64_t LEN_1D) {
    // Edge case: empty input
    if (LEN_1D <= 0) {
        // Undefined behavior; set result to NaN
        result[0] = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    // Query temporary storage size for reduction
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // First call to get required storage size
    CHECKHIP( hipcub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, a, result, static_cast<size_t>(LEN_1D)) );

    // Allocate temporary storage
    CHECKHIP( hipMalloc(&d_temp_storage, temp_storage_bytes) );

    // Perform the reduction
    CHECKHIP( hipcub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, a, result, static_cast<size_t>(LEN_1D)) );

    // Clean up temporary storage
    CHECKHIP( hipFree(d_temp_storage) );
    // Ensure kernel has completed before returning (optional)
    CHECKHIP( hipDeviceSynchronize() );
}

