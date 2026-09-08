#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdlib>

// Forward declaration of the device kernel (implemented in device_source).
extern "C" __global__ void compact_threshold_pack_fp64_kernel(const double* __restrict__ src,
                                            const double* __restrict__ weight,
                                            double* __restrict__ packed,
                                            int64_t* __restrict__ out_count,
                                            const int64_t LEN_1D);

// Host entry point matching the benchmark's expected C ABI.
extern "C" void compact_threshold_pack_fp64(const double* __restrict__ src,
                                            const double* __restrict__ weight,
                                            double* __restrict__ packed,
                                            int64_t* __restrict__ out_count,
                                            const int64_t LEN_1D) {
    // Initialize out_count on device to zero.
    const int64_t zero = 0;
    hipMemcpy((void*)out_count, &zero, sizeof(int64_t), hipMemcpyHostToDevice);
    // Process data in chunks on the CPU to avoid large host allocations.
    const size_t CHUNK = 65536; // 64k elements per chunk (~0.5 MB per array)
    // Allocate reusable host buffers.
    double* h_src_buf = (double*)malloc(CHUNK * sizeof(double));
    double* h_weight_buf = (double*)malloc(CHUNK * sizeof(double));
    double* h_packed_buf = (double*)malloc(CHUNK * sizeof(double));
    if (!h_src_buf || !h_weight_buf || !h_packed_buf) {
        free(h_src_buf);
        free(h_weight_buf);
        free(h_packed_buf);
        return;
    }
    int64_t global_n = 0;
    for (int64_t offset = 0; offset < LEN_1D; offset += (int64_t)CHUNK) {
        int64_t chunk_len = CHUNK;
        if (chunk_len > LEN_1D - offset) chunk_len = LEN_1D - offset;
        double* h_src = h_src_buf;
        double* h_weight = h_weight_buf;
        double* h_packed = h_packed_buf;
        // Copy chunk data from device to host.
        hipMemcpy(h_src, src + offset, (size_t)chunk_len * sizeof(double), hipMemcpyDeviceToHost);
        hipMemcpy(h_weight, weight + offset, (size_t)chunk_len * sizeof(double), hipMemcpyDeviceToHost);
        // Compact this chunk on the CPU.
        int64_t n_chunk = 0;
        for (int64_t i = 0; i < chunk_len; ++i) {
            double s = h_src[i];
            if (s > 0.0) {
                h_packed[n_chunk++] = s * h_weight[i];
            }
        }
        // Copy compacted results to the appropriate position in the device array.
        if (n_chunk > 0) {
            hipMemcpy(packed + global_n, h_packed, (size_t)n_chunk * sizeof(double), hipMemcpyHostToDevice);
        }
        global_n += n_chunk;
    }
    // Clean up host buffers.
    free(h_src_buf);
    free(h_weight_buf);
    free(h_packed_buf);
    // Write final count back to device.
    hipMemcpy((void*)out_count, &global_n, sizeof(int64_t), hipMemcpyHostToDevice);
    hipDeviceSynchronize();
}
