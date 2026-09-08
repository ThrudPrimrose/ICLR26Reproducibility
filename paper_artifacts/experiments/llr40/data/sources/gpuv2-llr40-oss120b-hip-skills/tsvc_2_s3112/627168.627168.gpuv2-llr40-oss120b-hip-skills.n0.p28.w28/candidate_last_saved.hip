#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <cstdint>
#include <cstdio>

inline void hip_check(hipError_t err, const char* file, int line) {
    if (err != hipSuccess) {
        fprintf(stderr, "HIP error at %s:%d: %s\n", file, line, hipGetErrorString(err));
        exit(1);
    }
}
#define HIP_CHECK(call) hip_check((call), __FILE__, __LINE__)

extern "C" void tsvc_2_s3112_fp64(const double* __restrict__ a, double* __restrict__ b, const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;
    const int num_items = static_cast<int>(LEN_1D);
    // Determine temporary storage size
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // Query the size of temporary storage needed for scan
    hipcub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, a, b, num_items);
    // Allocate temporary storage on device
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));
    // Perform inclusive scan (prefix sum)
    hipcub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, a, b, num_items);
    // Free temporary storage
    HIP_CHECK(hipFree(d_temp_storage));
    // Ensure all work is completed before returning
    HIP_CHECK(hipDeviceSynchronize());
}
