// v4: NUMA-aware distributed FDTD march.
//
// Each OpenMP thread owns a contiguous block of rows and marches on a local,
// 64B-aligned copy (first-touched on that thread's socket), with one extra row
// of halo on each side where the update expressions need it:
//   - ey update of row i needs hz rows i and i-1       -> hz halo on the left
//   - hz update of row i needs new ey row i+1          -> ey halo on the right
// Per step, only those two 1-row values cross threads, copied from the
// neighbour's local buffer around two barriers.  Every per-element expression
// is the reference's, so the FP rounding of each element is unchanged.
//
// All memory is allocated once on the calling thread (worker threads do no
// allocation) in one aligned block of 3 x nt x (Kmax+1) x NY doubles; slot
// addresses are deterministic, so no shared pointer tables are needed.
//
// Falls back to the plain fused loops for tiny work or a single thread.

#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>

/* count CPUs in the process affinity mask (cgroup cpuset + taskset);
   /proc/self/status line "Cpus_allowed: <hex>" — popcount of the nibbles */
static int fdtd_cpu_count(void) {
  static const int nib[16] = {0,1,1,2,1,2,1,2,1,2,1,2,1,2,1,2};
  FILE *fp = fopen("/proc/self/status", "r");
  if (!fp) return -1;
  char line[512];
  int total = -1;
  while (fgets(line, 512, fp)) {
    if (strncmp(line, "Cpus_allowed:", 13) == 0) {
      total = 0; char *p = line + 13;
      while (*p) { if (*p >= '0' && *p <= '9') total += nib[*p - '0'];
                   else if (*p >= 'a' && *p <= 'f') total += nib[*p - 'a' + 10];
                   p++; }
      break;
    }
  }
  fclose(fp);
  return total;
}

void fdtd_2d_fp64(double *restrict ex, double *restrict ey, const double *restrict fict, double *restrict hz, const int64_t NX, const int64_t NY, const int64_t TMAX, const double ex_courant, const double ey_courant, const double hz_courant) {
    const int64_t N = NX * NY;
    int nt_max = omp_get_max_threads();
    const int ncpu = fdtd_cpu_count();
    if (ncpu > 0 && ncpu < nt_max) nt_max = ncpu;
    if ((int)NX < nt_max) nt_max = (int)NX;

    if (!(nt_max >= 2 && TMAX >= 4 && N >= 8192)) {
      /* plain serial path */
      for (int64_t t = 0; t < TMAX; ++t) {
        const double f = fict[t];
        for (int64_t si0 = 0; si0 < NX; ++si0) {
          double *exr = ex + si0 * NY;
          double *eyr = ey + si0 * NY;
          const double *hzr = hz + si0 * NY;
          if (si0 == 0) {
            for (int64_t si1 = 0; si1 < NY; ++si1)
              eyr[si1] = f;
          } else {
            const double *hzp = hz + (si0 - 1) * NY;
            for (int64_t si1 = 0; si1 < NY; ++si1)
              eyr[si1] -= (ey_courant * (hzr[si1] - hzp[si1]));
          }
          for (int64_t si1 = 1; si1 < NY; ++si1)
            exr[si1] -= (ex_courant * (hzr[si1] - hzr[si1 - 1]));
        }
        for (int64_t si0 = 0; si0 < (NX - 1); ++si0) {
          double *exr = ex + si0 * NY;
          double *eyr = ey + si0 * NY;
          const double *eyr2 = ey + (si0 + 1) * NY;
          double *hzr = hz + si0 * NY;
          for (int64_t si1 = 0; si1 < (NY - 1); ++si1)
            hzr[si1] -= (hz_courant * (((exr[si1 + 1] - exr[si1]) + eyr2[si1]) - eyr[si1]));
        }
      }
      return;
    }

    const int64_t Kmax = (NX + nt_max - 1) / nt_max;
    const size_t slot = (size_t)(Kmax + 1) * (size_t)NY * 8;   /* bytes per row-slot */
    void *blk = aligned_alloc(64, 64 + (size_t)3 * (size_t)nt_max * slot);
    if (!blk) {
      /* allocation failure: plain serial path (duplicate, kept short) */
      for (int64_t t = 0; t < TMAX; ++t) {
        const double f = fict[t];
        for (int64_t si0 = 0; si0 < NX; ++si0) {
          double *exr = ex + si0 * NY;
          double *eyr = ey + si0 * NY;
          const double *hzr = hz + si0 * NY;
          if (si0 == 0) { for (int64_t si1 = 0; si1 < NY; ++si1) eyr[si1] = f; }
          else { const double *hzp = hz + (si0 - 1) * NY;
                 for (int64_t si1 = 0; si1 < NY; ++si1) eyr[si1] -= (ey_courant * (hzr[si1] - hzp[si1])); }
          for (int64_t si1 = 1; si1 < NY; ++si1)
            exr[si1] -= (ex_courant * (hzr[si1] - hzr[si1 - 1]));
        }
        for (int64_t si0 = 0; si0 < (NX - 1); ++si0) {
          double *exr = ex + si0 * NY;
          double *eyr = ey + si0 * NY;
          const double *eyr2 = ey + (si0 + 1) * NY;
          double *hzr = hz + si0 * NY;
          for (int64_t si1 = 0; si1 < (NY - 1); ++si1)
            hzr[si1] -= (hz_courant * (((exr[si1 + 1] - exr[si1]) + eyr2[si1]) - eyr[si1]));
        }
      }
      return;
    }
    double *base = (double *)blk + 8;   /* 8-double guard before the first row */

    _Pragma("omp parallel num_threads(nt_max)")
    {
      const int nt = omp_get_num_threads();
      const int me = omp_get_thread_num();
      const int64_t baseK = NX / nt, remK = NX % nt;
      const int64_t K = baseK + (me < remK ? 1 : 0);
      const int64_t st = baseK * me + (me < remK ? me : remK);
      const int64_t halo = (st > 0) ? 1 : 0;

      double *lex = base + (size_t)(me * 3 + 0) * slot / 8;
      double *ley = base + (size_t)(me * 3 + 1) * slot / 8;
      double *lhz = base + (size_t)(me * 3 + 2) * slot / 8;

      /* neighbour slots */
      const int64_t Kl = me > 0 ? (baseK + ((me - 1) < remK ? 1 : 0)) : 0;
      const int64_t stl = me > 0 ? (baseK * (me - 1) + ((me - 1) < remK ? me - 1 : remK)) : 0;
      const size_t hrowL = (size_t)(Kl - 1 + (stl > 0 ? 1 : 0)) * (size_t)NY;
      const double *lhzL = me > 0 ? base + (size_t)((me - 1) * 3 + 2) * slot / 8 : 0;
      const double *leyR = me + 1 < nt ? base + (size_t)((me + 1) * 3 + 1) * slot / 8 : 0;

      /* one-shot copy in */
      if (K > 0) {
        memcpy(lex, ex + st * NY, (size_t)K * (size_t)NY * 8);
        memcpy(ley, ey + st * NY, (size_t)K * (size_t)NY * 8);
      }
      if (st + K < NX)
        memcpy(ley + (size_t)K * (size_t)NY, ey + (st + K) * NY, (size_t)NY * 8);
      if (st > 0)
        memcpy(lhz, hz + (st - 1) * NY, (size_t)(K + 1) * (size_t)NY * 8);
      else if (K > 0)
        memcpy(lhz, hz + st * NY, (size_t)K * (size_t)NY * 8);

      const int64_t ib = st + K < NX - 1 ? st + K : NX - 1;  /* hz rows: [st, ib) */

      for (int64_t t = 0; t < TMAX; ++t) {
        const double f = fict[t];
        if (t > 0 && st > 0)
          memcpy(lhz, lhzL + hrowL, (size_t)NY * 8);

        /* phase A: ey boundary + ey/ex updates of my rows (local hz reads) */
        for (int64_t i = st; i < st + K; ++i) {
          double *exr = lex + (i - st) * NY;
          double *eyr = ley + (i - st) * NY;
          const double *hzr = lhz + (i - st + halo) * NY;
          if (i == 0) {
            for (int64_t si1 = 0; si1 < NY; ++si1)
              eyr[si1] = f;
          } else {
            const double *hzp = hzr - NY;
            for (int64_t si1 = 0; si1 < NY; ++si1)
              eyr[si1] -= (ey_courant * (hzr[si1] - hzp[si1]));
          }
          for (int64_t si1 = 1; si1 < NY; ++si1)
            exr[si1] -= (ex_courant * (hzr[si1] - hzr[si1 - 1]));
        }

        _Pragma("omp barrier")
        if (st + K < NX)
          memcpy(ley + (size_t)K * (size_t)NY, leyR, (size_t)NY * 8);

        /* phase B: hz update of my rows (new ex/ey) */
        for (int64_t i = st; i < ib; ++i) {
          double *exr = lex + (i - st) * NY;
          double *eyr = ley + (i - st) * NY;
          const double *eyr2 = (i + 1 < st + K) ? ley + (i + 1 - st) * NY : ley + (size_t)K * NY;
          double *hzr = lhz + (i - st + halo) * NY;
          for (int64_t si1 = 0; si1 < (NY - 1); ++si1)
            hzr[si1] -= (hz_courant * (((exr[si1 + 1] - exr[si1]) + eyr2[si1]) - eyr[si1]));
        }

        _Pragma("omp barrier")
      }

      /* one-shot copy out (my rows only) */
      if (K > 0) {
        memcpy(ex + st * NY, lex, (size_t)K * (size_t)NY * 8);
        memcpy(ey + st * NY, ley, (size_t)K * (size_t)NY * 8);
        memcpy(hz + st * NY, lhz + halo * NY, (size_t)K * (size_t)NY * 8);
      }
    }
    free(blk);
}
