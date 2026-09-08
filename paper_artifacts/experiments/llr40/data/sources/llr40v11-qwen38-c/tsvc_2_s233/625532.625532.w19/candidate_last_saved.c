/* TSVC tsvc_2_s233 -- two independent prefix scans:
 *   aa[j][i] = aa[j-1][i] + cc[j][i]   (serial in j, parallel over columns i)
 *   bb[j][i] = bb[j][i-1] + cc[j][i]   (serial in i, parallel over rows j)
 *
 * Bit-exact with the sequential reference: every element is produced by the
 * same left-to-right chain of fp64 additions.
 *
 * Scheduling (one fused OpenMP region):
 *   each thread owns (L/T) consecutive columns of the aa scan AND (L/T)
 *   consecutive rows of the bb scan.  The matrix is swept in 16-row slabs:
 *   every slab the thread updates its column slice for the slab's 16 rows
 *   (aa) and advances its row band by delta i values (bb).  All threads read
 *   the same 1.25 MB slab of cc, so each DRAM line is fetched once and the
 *   bb pass reuses it from L3; both output streams run concurrently.
 */
#include <stdint.h>
#include <stddef.h>
#include <omp.h>

void tsvc_2_s233_fp64(double *restrict aa, double *restrict bb,
                      const double *restrict cc, const int64_t LEN_2D)
{
  const int64_t N = LEN_2D;
  if (N <= 8) return;

  const int64_t L = N - 8;
  if (L < 64) { /* tiny: plain sequential, exactly like the reference */
    for (int64_t i = 8; i < N; ++i) {
      for (int64_t j = 8; j < N; ++j)
        aa[j * N + i] = aa[(j - 1) * N + i] + cc[j * N + i];
      for (int64_t j = 8; j < N; ++j)
        bb[j * N + i] = bb[j * N + (i - 1)] + cc[j * N + i];
    }
    return;
  }

  static int thr_set = 0;
  if (!thr_set) {
    thr_set = 1;
    int nt = omp_get_max_threads();
    if (nt > 96) nt = 96;
    if (nt < 1) nt = 1;
    omp_set_num_threads(nt);
  }

  #pragma omp parallel
  {
    const int64_t T = omp_get_num_threads();
    const int64_t t = omp_get_thread_num();

    const int64_t nslab = (L + 15) / 16; /* 16-row slabs */
    /* aa: columns owned by this thread (consecutive, 8-aligned start) */
    const int64_t col_start = t * ((L + T - 1) / T);
    const int64_t nc = L < col_start ? 0 : (L - col_start < (L + T - 1) / T ? L - col_start : (L + T - 1) / T);
    /* bb: rows owned by this thread (consecutive) */
    const int64_t row_start = t * ((L + T - 1) / T);
    const int64_t nr = L < row_start ? 0 : (L - row_start < (L + T - 1) / T ? L - row_start : (L + T - 1) / T);

    /* state */
    double *R = (double *)__builtin_alloca(nc > 0 ? nc * sizeof(double) : 1);
    const int64_t NROW = 48; /* bb rows interleaved */
    double c[NROW];
    const double *cpb[NROW];
    double *bpb[NROW];
    int64_t cur[NROW];
    for (int64_t r = 0; r < nr; r += NROW) {
      for (int64_t k = 0; k < NROW && r + k < nr; ++k) {
        const int64_t row = (row_start + r + k) * N;
        c[k] = bb[row + 7];
        cpb[k] = cc + row + 8;
        bpb[k] = bb + row + 8;
        cur[k] = 0;
      }
    }
    const int64_t ng = (nr + NROW - 1) / NROW;
    /* delta: i values each bb row advances per slab */
    const int64_t delta = (nr * L) / (nslab * NROW) + 1;
    const int64_t i0 = 8 + col_start;

    if (nc > 0) {
      for (int64_t c = 0; c < nc; ++c) R[c] = aa[7 * N + i0 + c];
    }

    for (int64_t s = 0; s < nslab; ++s) {
      const int64_t j0 = 8 + 16 * s;
      const int64_t nj = L - 16 * s < 16 ? L - 16 * s : 16;
      /* pass 1: my column slice for this slab's rows */
      if (nc > 0) {
        for (int64_t j = j0; j < j0 + nj; ++j) {
          const double *cp = cc + j * N + i0;
          double *ap = aa + j * N + i0;
          for (int64_t c = 0; c < nc; ++c) {
            R[c] += cp[c];
            ap[c] = R[c];
          }
        }
      }
      /* pass 2: advance my row band */
      for (int64_t g = 0; g < ng; ++g) {
        const int64_t rb = g * NROW;
        const int64_t gn = nr - rb < NROW ? nr - rb : NROW;
        for (int64_t k = 0; k < gn; ++k) {
          int64_t adv = L - cur[k] < delta ? L - cur[k] : delta;
          const double *cp = cpb[k] + cur[k];
          double *bp = bpb[k] + cur[k];
          double cv = c[k];
          for (int64_t a = 0; a < adv; ++a) {
            cv += cp[a];
            bp[a] = cv;
          }
          cur[k] += adv;
          c[k] = cv;
        }
      }
    }
  }
}
