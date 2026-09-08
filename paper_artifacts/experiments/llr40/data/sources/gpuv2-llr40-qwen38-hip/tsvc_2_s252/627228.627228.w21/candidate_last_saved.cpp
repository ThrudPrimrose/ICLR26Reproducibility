// Host half of the tsvc_2_s252 HIP submission.
// The device half (tsvc_2_s252.hip) provides the kernel and launcher.
#include <cstdint>

void tsvc_2_s252_launch(double* a, const double* b, const double* c, int64_t n);

extern "C" void tsvc_2_s252_fp64(double* a, const double* b, const double* c,
                                 int64_t LEN_1D, void* workspace,
                                 int64_t workspace_size) {
    (void)workspace;
    (void)workspace_size;
    tsvc_2_s252_launch(a, b, c, LEN_1D);
}
