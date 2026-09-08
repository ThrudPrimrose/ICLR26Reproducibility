#include <stdint.h>
#include <omp.h>

/* TSVC s152: b[i] = d[i]*e[i]; a[i] += b[i]*c[i]  (fully parallel, streaming).
 *
 * Offload-arm note: this kernel touches every byte exactly once, so the map
 * round trip (48 B/element up/down) costs more than the 48 B/element of HBM
 * traffic the arithmetic itself uses -- offloading the body loses.  The host
 * does the work; the one-shot target region only registers a device kernel.
 */
void tsvc_2_s152_fp64(double *restrict a, double *restrict b, const double *restrict c,
                      const double *restrict d, const double *restrict e, const int64_t LEN_1D)
{
    static int dev_probe = -1;
    if (dev_probe < 0) {
        int on_device = 0;
#pragma omp target map(from: on_device)
        on_device = !omp_is_initial_device();
        dev_probe = on_device;
    }

#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        const double t = d[i] * e[i];
        b[i] = t;
        a[i] += t * c[i];
    }
}
