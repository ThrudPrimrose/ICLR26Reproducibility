#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

namespace fst {
void launch(const double* a, double* out, long long n);
}

extern "C" void fuse_stencil_through_transient_fp64(const double* a, double* out, const long long LEN_1D) {
    fst::launch(a, out, LEN_1D);
}
