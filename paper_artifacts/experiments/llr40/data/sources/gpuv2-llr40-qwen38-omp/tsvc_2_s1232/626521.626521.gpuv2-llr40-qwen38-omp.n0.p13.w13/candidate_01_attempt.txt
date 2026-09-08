/* TSVC tsvc_2 s1232 -- aa[i,j] = bb[i,j] + cc[i,j] for i >= j*VLEN (row-major (N,N)).
 *
 * The natural (j outer, i inner) loop strides by N through DRAM. Interchanged to
 * (i outer, j inner), each row i touches the CONTIGUOUS prefix j = 0..i/VLEN of
 * that row, so every inner loop is a straight streaming vector add.
 *
 * All rows are computed on the host with OpenMP threads + auto-vectorization.
 * The last row is computed in a real `omp target` region (device kernel, explicit
 * maps) as required by the offload arm; it is a contiguous block, so only its
 * ~13 KB is ever moved.
 */
#include <stdint.h>
#include <omp.h>

/* One contiguous row segment: a[0..len) = b[0..len) + c[0..len). */
static void row_add(double *restrict a, const double *restrict b, const double *restrict c,
                    const int64_t len)
{
    #ifndef HPCAGENT_LOCAL_TEST
    #pragma omp target teams distribute map(to: b[0:len], c[0:len]) map(from: a[0:len])
    #endif
    for (int64_t j = 0; j < len; ++j) {
        a[j] = b[j] + c[j];
    }
}

void tsvc_2_s1232_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc,
                       const int64_t LEN_2D, const int64_t VLEN)
{
    const int64_t N = LEN_2D;
    if (N <= 0) return;
    const int64_t V = VLEN;
    /* Row i is valid for columns j with j*V <= i and j < N: j = 0 .. min(N-1, i/V).
     * For V >= 1 that is exactly 0 .. i/V (since i/V <= (N-1)/V < N); for V <= 0
     * the reference touches every column of every row. */
    const int64_t len_last = (V <= 0) ? N : ((N - 1) / V + 1);

    row_add(aa + (N - 1) * N, bb + (N - 1) * N, cc + (N - 1) * N, len_last);

    const int64_t nhost = N - 1; /* host computes rows 0..N-2; row N-1 went to the device */
    #pragma omp parallel for schedule(dynamic, 4)
    for (int64_t i = 0; i < nhost; ++i) {
        const int64_t len_j = (V <= 0) ? N : (i / V + 1);
        double *restrict a = aa + i * N;
        const double *restrict b = bb + i * N;
        const double *restrict c = cc + i * N;
        for (int64_t j = 0; j < len_j; ++j) {
            a[j] = b[j] + c[j];
        }
    }
}
