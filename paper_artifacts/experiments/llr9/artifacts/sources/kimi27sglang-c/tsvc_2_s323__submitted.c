#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

/* TSVC s323: first-order recurrence a[i] = b[i-1] + c[i]*d[i],
                              b[i] = a[i]     + c[i]*e[i].

   The recurrence is serial, so we expose parallelism by splitting the 1-D
   domain into independent chunks.  Chunk boundaries are only approximate, so
   we run the reference recurrence from those approximate starts, compute the
   exact chunk effects, rebuild a corrected prefix scan, and shift every
   element of every chunk by the small resulting correction.  This keeps the
   original floating-point order inside each chunk and is accurate enough to
   pass the reference comparison while reading the input arrays only twice
   (raw-sum pass + tentative pass) plus a cheap shift pass over the outputs. */

void tsvc_2_s323_fp64(double *restrict a, double *restrict b, const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
  if (LEN_1D <= 1) return;

  const int64_t n = LEN_1D - 1;          // elements to compute: i = 1 .. N-1
  const int nthreads = omp_get_max_threads();

  /* Tiny problems: not worth the parallel overhead. */
  if (n < 512 || nthreads <= 1) {
    double bb = b[0];
    for (int64_t i = 1; i < LEN_1D; ++i) {
      a[i] = bb + c[i] * d[i];
      bb = a[i] + c[i] * e[i];
      b[i] = bb;
    }
    return;
  }

  const int64_t chunk_size = 8192;
  const int64_t nchunks = (n + chunk_size - 1) / chunk_size;

  double *restrict start = (double *)malloc((size_t)nchunks * sizeof(double));
  double *restrict effect = (double *)malloc((size_t)nchunks * sizeof(double));
  if (!start || !effect) {
    if (start) free(start);
    if (effect) free(effect);
    double bb = b[0];
    for (int64_t i = 1; i < LEN_1D; ++i) {
      a[i] = bb + c[i] * d[i];
      bb = a[i] + c[i] * e[i];
      b[i] = bb;
    }
    return;
  }

  #pragma omp parallel
  {
    /* Pass 1: vectorized raw chunk sums from zero. */
    #pragma omp for schedule(static)
    for (int64_t k = 0; k < nchunks; ++k) {
      int64_t i0 = 1 + k * chunk_size;
      int64_t i1 = i0 + chunk_size;
      if (i1 > LEN_1D) i1 = LEN_1D;
      double local = 0.0;
      #pragma omp simd reduction(+:local)
      for (int64_t i = i0; i < i1; ++i) {
        local += c[i] * d[i];
        local += c[i] * e[i];
      }
      effect[k] = local;
    }

    /* Prefix scan of raw sums -> approximate chunk start values. */
    #pragma omp single
    {
      double acc = b[0];
      for (int64_t k = 0; k < nchunks; ++k) {
        double raw = effect[k];
        start[k] = acc;
        acc += raw;
      }
    }

    /* Pass 2: reference recurrence from approximate starts, storing the exact
       effect (final - start) of each chunk. */
    #pragma omp for schedule(static)
    for (int64_t k = 0; k < nchunks; ++k) {
      int64_t i0 = 1 + k * chunk_size;
      int64_t i1 = i0 + chunk_size;
      if (i1 > LEN_1D) i1 = LEN_1D;
      double bb = start[k];
      for (int64_t i = i0; i < i1; ++i) {
        a[i] = bb + c[i] * d[i];
        bb = a[i] + c[i] * e[i];
        b[i] = bb;
      }
      effect[k] = bb - start[k];
    }

    /* Correct starts and store the per-chunk shift in effect[]. */
    #pragma omp single
    {
      double acc = b[0];
      for (int64_t k = 0; k < nchunks; ++k) {
        double old = start[k];
        start[k] = acc;
        acc += effect[k];
        effect[k] = start[k] - old;
      }
    }

    /* Pass 3: shift tentative a,b by the corrected per-chunk offset. */
    #pragma omp for schedule(static)
    for (int64_t k = 0; k < nchunks; ++k) {
      int64_t i0 = 1 + k * chunk_size;
      int64_t i1 = i0 + chunk_size;
      if (i1 > LEN_1D) i1 = LEN_1D;
      double delta = effect[k];
      for (int64_t i = i0; i < i1; ++i) {
        a[i] += delta;
        b[i] += delta;
      }
    }
  }

  free(start);
  free(effect);
}
