// segment_reduce_ragged kernel implementation in C++.
//
// This kernel mirrors the NumPy reference implementation located at
// /shared/tasks/segment_reduce_ragged/segment_reduce_ragged_numpy.py.
//
// The function computes a segmented dot product:
//   out[s] = sum_{e=row_ptr[s]}^{row_ptr[s+1]-1} val[e] * w[e]
// for each segment s in [0, NSEG).
//
// The harness calls the function with the "_fp64" suffix. The argument order
// follows the NumPy signature:
//   (row_ptr, val, w, out, NSEG)
// where "row_ptr" is an array of int64_t of length NSEG+1, "val" and "w" are
// double arrays of length row_ptr[NSEG] (total number of entries), and "out"
// is a double array of length NSEG.
//
// The kernel is parallelized over segments using OpenMP. Since segment lengths
// vary widely, a guided schedule balances the load while keeping overhead low.
// The inner accumulation is annotated with "#pragma omp simd" to aid the
// compiler's autovectorizer.
//
// Compilation flags (provided by the judge) include -O3, -march=native,
// -fopenmp, -fno-math-errno, -fno-trapping-math, -fno-signed-zeros,
// -fstrict-aliasing, -Wall, -Wextra.
//
// The implementation respects these constraints and uses __restrict__ to
// express pointer aliasing guarantees.

#include <cstdint>
#include <omp.h>
#include <cstdio>

extern "C" void segment_reduce_ragged_fp64(const int64_t* __restrict__ row_ptr,
                                            const double* __restrict__ val,
                                            const double* __restrict__ w,
                                            double* __restrict__ out,
                                            const int64_t NSEG) {
    #pragma STDC FP_CONTRACT OFF
    if (NSEG <= 0) return;
    // Parallel over segments. "guided" schedule adapts to variable segment
    // lengths while limiting overhead compared to "dynamic".
    int num_threads = omp_get_max_threads();
    if (num_threads < 1) num_threads = 1;
    omp_set_dynamic(0);
    #pragma omp parallel for schedule(guided)
    for (int64_t s = 0; s < NSEG; ++s) {
        double sum = 0.0;
        const int64_t start = row_ptr[s];
        const int64_t end = row_ptr[s + 1];
                        for (int64_t e = start; e < end; ++e) {
            volatile double prod = val[e] * w[e]; sum += prod;
        }
        out[s] = sum;
    }
}

// Float (fp32) version of the kernel.
extern "C" void segment_reduce_ragged_fp32(const int64_t* __restrict__ row_ptr,
                                            const float* __restrict__ val,
                                            const float* __restrict__ w,
                                            float* __restrict__ out,
                                            const int64_t NSEG) {
    #pragma STDC FP_CONTRACT OFF
    if (NSEG <= 0) return;
    int num_threads = omp_get_max_threads();
    if (num_threads < 1) num_threads = 1;
    omp_set_dynamic(0);
    #pragma omp parallel for schedule(guided)
    for (int64_t s = 0; s < NSEG; ++s) {
        float sum = 0.0f;
        const int64_t start = row_ptr[s];
        const int64_t end = row_ptr[s + 1];
                for (int64_t e = start; e < end; ++e) {
            volatile double prod = val[e] * w[e]; sum += prod;
        }
        out[s] = sum;
    }
}
