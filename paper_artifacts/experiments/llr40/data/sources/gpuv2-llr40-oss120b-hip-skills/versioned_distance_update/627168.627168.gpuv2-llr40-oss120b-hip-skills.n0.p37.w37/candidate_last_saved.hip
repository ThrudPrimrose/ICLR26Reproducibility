#include <hip/hip_runtime.h>
#include <cstdint>
#include <hipcub/hipcub.hpp>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(call) do { hipError_t err = (call); if (err != hipSuccess) { \
    fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
    std::exit(EXIT_FAILURE); } } while (0)

// Pair struct for K=1 prefix-scan
struct Pair {
    double mul;
    double add;
};

struct PairOp {
    __host__ __device__
    Pair operator()(const Pair &a, const Pair &b) const {
        Pair r;
        r.mul = a.mul * b.mul;
        r.add = a.add * b.mul + b.add;
        return r;
    }
};

// Forward declaration of the device kernel.
extern "C" __global__
void versioned_distance_update_kernel(double * __restrict__ a,
                                      const double * __restrict__ b,
                                      const double * __restrict__ c,
                                      const int64_t K,
                                      const int64_t LEN_1D);

extern "C" __global__
void init_tmp_kernel(double * __restrict__ a,
                     const double * __restrict__ b,
                     const double * __restrict__ c,
                     double * __restrict__ tmp,
                     const int64_t len);
extern "C" __global__
void finalize_kernel(double * __restrict__ a,
                     const double * __restrict__ tmp,
                     const int64_t len);

extern "C" __global__
void fill_pair_kernel(const double * __restrict__ b,
                      const double * __restrict__ c,
                      Pair * __restrict__ pairs,
                      const int64_t len);
extern "C" __global__
void apply_scan_kernel(double * __restrict__ a,
                       const Pair * __restrict__ pairs,
                       const int64_t len);

extern "C" void versioned_distance_update_fp64(double * __restrict__ a,
                                                const double * __restrict__ b,
                                                const double * __restrict__ c,
                                                const int64_t LEN_1D,
                                                const int64_t K) {
    // Determine launch configuration: compute actual K and LEN values.
    // Some manifests may pass arguments in either order; swap if needed.
    int64_t len = LEN_1D;
    int64_t k = K;
    if (k > len) {
        // Likely arguments were passed as (K, LEN) instead of (LEN, K)
        int64_t tmp = len;
        len = k;
        k = tmp;
    }
    const int threads_per_block = 256;
    // Ensure we have at least one thread per chain (up to k).
    int64_t blocks = (k + threads_per_block - 1) / threads_per_block;
    if (blocks < 1) blocks = 1;
    if (k == 1) {
        // Special case: use a parallel prefix scan with custom operator
        // Allocate array of Pair structs (size len-1)
        Pair* pairs;
        HIP_CHECK(hipMalloc(&pairs, (len - 1) * sizeof(Pair)));
        // Fill pairs with mul=0.75 and add=b[i]*c[i]
        int64_t pair_blocks = (len - 1 + threads_per_block - 1) / threads_per_block;
        hipLaunchKernelGGL(fill_pair_kernel,
                           dim3(pair_blocks), dim3(threads_per_block),
                           0, 0,
                           b, c, pairs, len);
        HIP_CHECK(hipGetLastError());
        // Inclusive scan over pairs using custom operator
        size_t temp_storage_bytes = 0;
        void* d_temp_storage = nullptr;
        hipError_t scan_err1 = hipcub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, pairs, pairs, PairOp(), len - 1);
        HIP_CHECK(scan_err1);
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));
        hipError_t scan_err2 = hipcub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, pairs, pairs, PairOp(), len - 1);
        HIP_CHECK(scan_err2);
        hipFree(d_temp_storage);
        // Apply scan results to compute final a values (skip i=0)
        hipLaunchKernelGGL(apply_scan_kernel,
                           dim3(pair_blocks), dim3(threads_per_block),
                           0, 0,
                           a, pairs, len);
        HIP_CHECK(hipGetLastError());
        hipFree(pairs);
        HIP_CHECK(hipDeviceSynchronize());
        return;
    }
    // Launch the device kernel with corrected k and len order.
    hipLaunchKernelGGL(versioned_distance_update_kernel,
                       dim3(static_cast<unsigned int>(blocks)),
                       dim3(threads_per_block),
                       0, 0,
                       a, b, c, k, len);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
}
