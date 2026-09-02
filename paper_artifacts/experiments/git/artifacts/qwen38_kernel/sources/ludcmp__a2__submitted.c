#include <stdint.h>
#include <immintrin.h>

/* LU factorization without pivoting (Doolittle, right-looking) + triangular
 * solves.  A is factorized in place: strictly-lower part = L (unit diagonal),
 * upper part = U.  y solves L y = b, x solves U x = y.
 *
 * Rounding must match the NumPy reference elementwise:
 *   A[i,j] -= A[i,k]*A[k,j]   (separate rounded multiply, then subtract)
 *   A[i,k]  /= A[k,k]         (true division)
 * The default -ffp-contract=fast fuses "mul then sub" into an FMA, so the
 * product is computed with an explicit FMA whose addend is exactly zero:
 *   fl(u*l + 0) == fl(u*l)  bit for bit (adding +0 to the exact product
 * changes nothing, so the single FMA rounding equals the multiply rounding),
 * and an explicit FMA intrinsic cannot itself be contracted with the
 * following subtract.  The l > 0 guard selects it only when a zero FMA
 * addend cannot flip the sign bit of a zero product.
 *
 * The 8-lane group loop uses AVX-512 intrinsics over full 8-wide groups; the
 * tail uses 8-byte (sd) loads/stores only -- a 16-byte access there would
 * touch element j+1, which for a row's last column is the *next* row's L
 * element and racy with that row's division.
 * One persistent OpenMP team is forked for the whole factorization; each
 * step k is a workshare over rows (a barrier per step, no per-step team
 * forks).  The triangular solves stay in a separate noinline function, away
 * from the OpenMP region.
 */

static __attribute__((noinline)) void lu_all(double *restrict A, int64_t N) {
    const int64_t ld = N;
    #pragma omp parallel
    {
        for (int64_t k = 0; k < N; ++k) {
            const double pivot = A[k * ld + k];
            const double *urow = A + k * ld + k + 1;
            const int64_t m = N - k - 1; /* columns k+1..N-1 */
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < m; ++i) {
                double *rowp = A + (k + 1 + i) * ld + k; /* L element, then row */
                double l = *rowp /= pivot;
                double *row = rowp + 1;
                int64_t j = 0;
                if (l > 0.0) {
                    const __m512d lv = _mm512_set1_pd(l);
                    const __m512d z = _mm512_setzero_pd();
                    for (; j + 8 <= m; j += 8) {
                        __m512d r = _mm512_loadu_pd(row + j);
                        __m512d p = _mm512_fmadd_pd(_mm512_loadu_pd(urow + j), lv, z);
                        _mm512_storeu_pd(row + j, _mm512_sub_pd(r, p));
                    }
                    const __m128d ls = _mm_set_sd(l);
                    const __m128d zs = _mm_setzero_pd();
                    for (; j < m; ++j) {
                        __m128d r = _mm_load_sd(row + j);
                        __m128d p = _mm_fmadd_sd(_mm_load_sd(urow + j), ls, zs);
                        _mm_store_sd(row + j, _mm_sub_sd(r, p));
                    }
                } else {
                    /* cold: two-step rounding, no FMA can form (also safe for
                     * l == 0, where a zero FMA addend would flip the sign bit
                     * of negative-zero products). */
                    for (; j < m; ++j) {
                        volatile double p = urow[j] * l;
                        row[j] -= p;
                    }
                }
            }
        }
    }
}

static __attribute__((noinline)) void tri_solves(double *restrict A,
                                                 const double *restrict b,
                                                 double *restrict x,
                                                 double *restrict y, int64_t N) {
    const int64_t ld = N;
    /* forward substitution: y[i] = b[i] - sum_{j<i} A[i,j]*y[j] */
    for (int64_t i = 0; i < N; ++i) {
        double s = 0.0;
        const double *row = A + i * ld;
        for (int64_t j = 0; j < i; ++j)
            s += row[j] * y[j];
        y[i] = b[i] - s;
    }
    /* back substitution: x[i] = (y[i] - sum_{j>i} A[i,j]*x[j]) / A[i,i] */
    for (int64_t i = N - 1; i >= 0; --i) {
        double s = 0.0;
        const double *row = A + i * ld + i + 1;
        for (int64_t j = i + 1; j < N; ++j)
            s += row[j - (i + 1)] * x[j];
        x[i] = (y[i] - s) / A[i * ld + i];
    }
}

void ludcmp_fp64(double *restrict A, const double *restrict b,
                 double *restrict x, double *restrict y, int64_t N) {
    if (N > 0) {
        lu_all(A, N);
        tri_solves(A, b, x, y, N);
    }
}
