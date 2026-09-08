#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

extern "C" void argmax_with_index_fp64(const double *__restrict__ a,
                                         int64_t *__restrict__ out_index,
                                         double *__restrict__ out_value,
                                         const int64_t LEN_1D) {
    if (LEN_1D <= 0) {
        return;
    }
    // Determine temporary storage size
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    hipError_t err = hipcub::DeviceReduce::ArgMax(d_temp_storage,
                                                  temp_storage_bytes,
                                                  a,
                                                  out_value,
                                                  out_index,
                                                  static_cast<size_t>(LEN_1D),
                                                  0);
    if (err != hipSuccess) {
        abort();
    }
    // Allocate temporary storage
    err = hipMalloc(&d_temp_storage, temp_storage_bytes);
    if (err != hipSuccess) {
        abort();
    }
    // Actual reduction
    err = hipcub::DeviceReduce::ArgMax(d_temp_storage,
                                      temp_storage_bytes,
                                      a,
                                      out_value,
                                      out_index,
                                      static_cast<size_t>(LEN_1D),
                                      0);
    if (err != hipSuccess) {
        hipFree(d_temp_storage);
        abort();
    }
    // Free temporary storage
    hipFree(d_temp_storage);
}
