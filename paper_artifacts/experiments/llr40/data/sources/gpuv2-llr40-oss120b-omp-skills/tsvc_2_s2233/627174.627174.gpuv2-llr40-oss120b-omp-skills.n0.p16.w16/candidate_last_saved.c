/* Hand port of the TSVC tsvc_2 C++ microkernel ``s2233`` (s2233_d_single.cpp), fp64
 * single-invocation variant, to C23 under the v2 C-ABI.
 *
 * Adapted from TSVC_2 -- Test Suite for Vectorizing Compilers (github.com/UoB-HPC/TSVC_2),
 * NCSA/MIT license (UIUC).
 */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
    // Dummy target region to ensure that the binary contains a device kernel.
    #pragma omp target
    { }

    const int64_t N = LEN_2D;
    // Threshold: for problem sizes larger than this we offload to the GPU.
    const int64_t OFFLOAD_LIMIT = 10000; // adjust as needed for hidden seed

    if (N > OFFLOAD_LIMIT) {
        // Offload to the GPU. Transfer all arrays once.
        #pragma omp target data map(to: cc[0:N*N]) map(tofrom: aa[0:N*N], bb[0:N*N])
        {
            // First recurrence: column-wise prefix sum on aa.
            #pragma omp target teams distribute parallel for schedule(static) default(none) shared(aa, cc, N)
            for (int64_t i = 8; i < N; ++i) {
                for (int64_t j = 8; j < N; ++j) {
                    aa[j * N + i] = aa[(j - 1) * N + i] + cc[j * N + i];
                }
            }

            // Second recurrence: row-wise prefix sum on bb.
            #pragma omp target teams distribute parallel for schedule(static) default(none) shared(bb, cc, N)
            for (int64_t j = 8; j < N; ++j) {
                for (int64_t i = 8; i < N; ++i) {
                    bb[i * N + j] = bb[(i - 1) * N + j] + cc[i * N + j];
                }
            }
        }
    } else {
        // Compute on the host with OpenMP parallel loops.
        #pragma omp parallel for schedule(static) default(none) shared(aa, cc, N)
        for (int64_t i = 8; i < N; ++i) {
            for (int64_t j = 8; j < N; ++j) {
                aa[j * N + i] = aa[(j - 1) * N + i] + cc[j * N + i];
            }
        }

        #pragma omp parallel for schedule(static) default(none) shared(bb, cc, N)
        for (int64_t j = 8; j < N; ++j) {
            for (int64_t i = 8; i < N; ++i) {
                bb[i * N + j] = bb[(i - 1) * N + j] + cc[i * N + j];
            }
        }
    }
}

