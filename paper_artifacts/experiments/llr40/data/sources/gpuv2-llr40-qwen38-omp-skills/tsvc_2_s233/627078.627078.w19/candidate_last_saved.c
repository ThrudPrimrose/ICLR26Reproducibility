/* tsvc_2 s233:
 *   aa[j][i] = aa[j-1][i] + cc[j][i]   (j,i >= 8): column-wise prefix scan, columns independent
 *   bb[j][i] = bb[j][i-1] + cc[j][i]   (j,i >= 8): row-wise prefix scan, rows independent
 * Each scan stays in exact left-to-right order (bit-identical to the reference);
 * the scans are distributed over threads. A thread owns 8 consecutive columns of aa
 * (8 independent chains -> one 512-bit vector) or 4 consecutive rows of bb (ILP). */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s233_fp64(double *restrict aa, double *restrict bb,
                      const double *restrict cc, const int64_t n)
{
  if (n <= 8)
    return;

  #pragma omp parallel
  {
    /* ---------------- aa: column-wise scans, 8 columns per chunk ---------------- */
    const int64_t ncol = n - 8;
    const int64_t full_i = 8 + 8 * (ncol / 8); /* first col of partial tail */

    #pragma omp for schedule(static)
    for (int64_t i0 = 8; i0 < full_i; i0 += 8)
    {
      double v0, v1, v2, v3, v4, v5, v6, v7;
      v0 = aa[7 * n + i0 + 0];
      v1 = aa[7 * n + i0 + 1];
      v2 = aa[7 * n + i0 + 2];
      v3 = aa[7 * n + i0 + 3];
      v4 = aa[7 * n + i0 + 4];
      v5 = aa[7 * n + i0 + 5];
      v6 = aa[7 * n + i0 + 6];
      v7 = aa[7 * n + i0 + 7];
      for (int64_t j = 8; j < n; ++j)
      {
        v0 += cc[j * n + i0 + 0];
        aa[j * n + i0 + 0] = v0;
        v1 += cc[j * n + i0 + 1];
        aa[j * n + i0 + 1] = v1;
        v2 += cc[j * n + i0 + 2];
        aa[j * n + i0 + 2] = v2;
        v3 += cc[j * n + i0 + 3];
        aa[j * n + i0 + 3] = v3;
        v4 += cc[j * n + i0 + 4];
        aa[j * n + i0 + 4] = v4;
        v5 += cc[j * n + i0 + 5];
        aa[j * n + i0 + 5] = v5;
        v6 += cc[j * n + i0 + 6];
        aa[j * n + i0 + 6] = v6;
        v7 += cc[j * n + i0 + 7];
        aa[j * n + i0 + 7] = v7;
      }
    }
    #pragma omp for schedule(static)
    for (int64_t i = full_i; i < n; ++i)
    {
      double v = aa[7 * n + i];
      for (int64_t j = 8; j < n; ++j)
      {
        v += cc[j * n + i];
        aa[j * n + i] = v;
      }
    }

    /* ---------------- bb: row-wise scans, 4 rows per chunk (ILP) ---------------- */
    const int64_t nrow = n - 8;
    const int64_t full_j = 8 + 4 * (nrow / 4); /* first row of partial tail */

    #pragma omp for schedule(static)
    for (int64_t j0 = 8; j0 < full_j; j0 += 4)
    {
      double w0 = bb[j0 * n + 7];
      double w1 = bb[(j0 + 1) * n + 7];
      double w2 = bb[(j0 + 2) * n + 7];
      double w3 = bb[(j0 + 3) * n + 7];
      for (int64_t i = 8; i < n; ++i)
      {
        w0 += cc[j0 * n + i];
        bb[j0 * n + i] = w0;
        w1 += cc[(j0 + 1) * n + i];
        bb[(j0 + 1) * n + i] = w1;
        w2 += cc[(j0 + 2) * n + i];
        bb[(j0 + 2) * n + i] = w2;
        w3 += cc[(j0 + 3) * n + i];
        bb[(j0 + 3) * n + i] = w3;
      }
    }
    #pragma omp for schedule(static)
    for (int64_t j = full_j; j < n; ++j)
    {
      double w = bb[j * n + 7];
      for (int64_t i = 8; i < n; ++i)
      {
        w += cc[j * n + i];
        bb[j * n + i] = w;
      }
    }
  }
}
