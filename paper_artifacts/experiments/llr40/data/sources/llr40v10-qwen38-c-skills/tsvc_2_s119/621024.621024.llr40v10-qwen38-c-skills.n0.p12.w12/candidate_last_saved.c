/* TSVC tsvc_2 s119: aa[i][j] = aa[i-1][j-1] + bb[i][j],  i,j in [1, n).
 *
 * Dependence: (1,1).  The matrix is split into wide column strips of width W;
 * each strip is row-blocked in R.  A thread owns whole strips; inside a strip
 * the row blocks run top to bottom, so the horizontal read of the left
 * strip's last column is always satisfied in-thread.  Across threads, strip
 * tile (r,s) needs only the last row of tiles (r-1,s) and (r-1,s-1): one
 * flag pair per row block, so the grid pipelines without global barriers.
 * Rows inside a tile are unit stride and SIMD-able; every line is fully used.
 */
#include <stdint.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *flag_buf = NULL;
static size_t flag_cap = 0;

void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D)
{
    const int64_t n = LEN_2D;
    if (n < 2) return;

    const int64_t T = (int64_t)omp_get_max_threads();

    /* ~T/2 strips so each thread owns about two (independent) strips. */
    int64_t W = 2 * (n - 1) / T;
    if (W < 8) W = 8;
    if (W > 4096) W = 4096;
    const int64_t nt_s = (n - 1 + W - 1) / W;      /* strips */
    int64_t R = (n - 1) / nt_s;
    if (R < 8) R = 8;
    if (R > 8192) R = 8192;
    const int64_t nt_r = (n - 1 + R - 1) / R;      /* row blocks per strip */

    const size_t nflags = (size_t)nt_s * nt_r;
    if (nflags > flag_cap) {
        if (flag_buf) free(flag_buf);
        flag_cap = nflags < 262144 ? 262144 : nflags;
        flag_buf = (unsigned char *)aligned_alloc(64, flag_cap);
    }
    memset(flag_buf, 0, nflags);

    #pragma omp parallel
    {
        const int64_t tid = (int64_t)omp_get_thread_num();
        const int64_t TT = (int64_t)omp_get_num_threads();
        const int64_t s0 = (int64_t)(((uint64_t)tid * (uint64_t)nt_s) / (uint64_t)TT);
        const int64_t s1 = (int64_t)(((uint64_t)(tid + 1) * (uint64_t)nt_s) / (uint64_t)TT);
        for (int64_t r = 0; r < nt_r; ++r) {
            const int64_t i0 = 1 + r * R;
            int64_t i1 = i0 + R;
            if (i1 > n) i1 = n;
            const unsigned char *pf = (r > 0) ? flag_buf + (size_t)(r - 1) * nt_s : NULL;
            for (int64_t s = s0; s < s1; ++s) {
                if (pf) {
                    long spins = 0;
                    while (__atomic_load_n(&pf[s], __ATOMIC_ACQUIRE) == 0) {
                        if (++spins > 256) __builtin_ia32_pause();
                    }
                    if (s > 0) {
                        spins = 0;
                        while (__atomic_load_n(&pf[s - 1], __ATOMIC_ACQUIRE) == 0) {
                            if (++spins > 256) __builtin_ia32_pause();
                        }
                    }
                }
                const int64_t j0 = 1 + s * W;
                int64_t j1 = j0 + W;
                if (j1 > n) j1 = n;
                for (int64_t i = i0; i < i1; ++i) {
                    double *restrict row = aa + i * n;
                    const double *restrict prev = row - n;
                    const double *restrict brr = bb + i * n;
                    for (int64_t j = j0; j < j1; ++j)
                        row[j] = prev[j - 1] + brr[j];
                }
                __atomic_store_n(&flag_buf[(size_t)r * nt_s + s], 1, __ATOMIC_RELEASE);
            }
        }
    }
}
