#include <hip/hip_runtime.h>
#include <cstdint>

void tsvc2s319_launch(double* a, double* b, const double* c, const double* d, const double* e,
                      const int64_t n, double* ws);

static double* g_ws = nullptr;

extern "C" void tsvc_2_s319_fp64(double* a, double* b, const double* c, const double* d,
                                 const double* e, const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;
    if (!g_ws) {
        hipError_t err = hipMalloc(&g_ws, (1u << 20) * sizeof(double));
        if (err != hipSuccess) { g_ws = nullptr; return; }
        hipMemset(g_ws, 0, (1u << 20) * sizeof(double));
    }
    tsvc2s319_launch(a, b, c, d, e, LEN_1D, g_ws);
}
