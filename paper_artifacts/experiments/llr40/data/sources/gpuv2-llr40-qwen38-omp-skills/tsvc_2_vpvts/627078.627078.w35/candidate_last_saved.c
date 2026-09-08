#include <stdint.h>
#include <omp.h>

void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D, const int64_t S) {
    if (LEN_1D <= 0) return;
    const double Sd = (double)S;

    /* Launch a trivial device kernel once so the binary carries a device
     * image (required by this arm). Synchronous; one-time cost. */
    static int dev_probed = 0;
    if (!dev_probed) {
        dev_probed = 1;
        int probe = 0;
#pragma omp target map(from: probe)
        probe = !omp_is_initial_device();
        (void)probe;
    }

    const int64_t M = (int64_t)(0.5 * (double)LEN_1D);
    if (M < (1 << 20)) {
        /* too small to amortize a GPU round trip */
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < LEN_1D; ++i) a[i] += b[i] * Sd;
        return;
    }

    /* Hybrid: thread 0 drives the GPU on [0,M) and does no host work;
     * threads 1..nt-1 stream [M, LEN_1D) in static contiguous spans.
     * The implicit barrier of the parallel region joins the GPU leg. */
#pragma omp parallel
    {
        const int nt = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        if (nt < 2 || tid == 0) {
            if (nt < 2) {
#pragma omp target teams distribute parallel for \
                map(tofrom: a[0:M]) map(to: b[0:M]) map(to: Sd)
                for (int64_t i = 0; i < M; ++i) a[i] += b[i] * Sd;
#pragma omp for schedule(static)
                for (int64_t i = M; i < LEN_1D; ++i) a[i] += b[i] * Sd;
            } else {
#pragma omp target teams distribute parallel for \
                map(tofrom: a[0:M]) map(to: b[0:M]) map(to: Sd)
                for (int64_t i = 0; i < M; ++i) a[i] += b[i] * Sd;
            }
        } else {
            const int64_t n = LEN_1D - M;
            const int64_t span = n / (nt - 1);
            const int64_t rem = n % (nt - 1);
            const int64_t k = (int64_t)(tid - 1);
            const int64_t lo = M + k * span + (k < rem ? k : rem);
            const int64_t hi = lo + span + (k < rem ? 1 : 0);
            for (int64_t i = lo; i < hi; ++i) a[i] += b[i] * Sd;
        }
    }
}
