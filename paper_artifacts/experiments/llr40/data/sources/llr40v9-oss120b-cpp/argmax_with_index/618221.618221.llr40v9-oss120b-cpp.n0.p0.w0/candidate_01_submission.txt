#include <cstdint>
#include <limits>

extern "C" void argmax_with_index_fp64(const double* __restrict__ a, int64_t* __restrict__ out_index, double* __restrict__ out_value, const int64_t LEN_1D) {
    struct MaxIdx {
        double val;
        int64_t idx;
    };
    #pragma omp declare reduction(maxloc : MaxIdx : omp_out = (omp_in.val > omp_out.val) || (omp_in.val == omp_out.val && omp_in.idx < omp_out.idx) ? omp_in : omp_out) initializer(omp_priv = { -std::numeric_limits<double>::infinity(), INT64_MAX })
    MaxIdx result = { -std::numeric_limits<double>::infinity(), INT64_MAX };
    #pragma omp parallel for reduction(maxloc:result)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        MaxIdx cand = { a[i], i };
        if (cand.val > result.val || (cand.val == result.val && cand.idx < result.idx)) {
            result = cand;
        }
    }
    out_value[0] = result.val;
    out_index[0] = result.idx;
}
