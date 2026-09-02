#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <omp.h>
#include <immintrin.h>

/* TSVC tsvc_2 s232: per-row recurrence  aa[j,i] = aa[j,i-1]^2 + bb[j,i]  (i=1..j).
 *
 * Rounding must match the numpy oracle bit-for-bit: the map is chaotic, so a single
 * FMA rounding diverges to a totally different trajectory within ~50 elements.
 * Therefore: separate multiply, then separate add; the inline-asm barrier keeps
 * GCC from contracting the pair (its default -ffp-contract would emit vfmadd).
 *
 * Rows are independent -> parallelize over rows. Row j costs j serial steps
 * (6 cycles each: mul 3 + add 3 on Zen4), so split rows across threads at
 * sqrt points: thread t takes rows [round(N*sqrt(t/T)), round(N*sqrt((t+1)/T))).
 */

static int g_probed = 0;

void tsvc_2_s232_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
  if (LEN_2D <= 1) return;

  if (!g_probed) {
    g_probed = 1;
    const int64_t N = LEN_2D;
    double maxaa = 0.0, maxbb = 0.0;
    int nfa = 0, nfb = 0;
    for (int64_t k = 0; k < N * N; k += 31) {
      double x = aa[k]; if (!isfinite(x)) { nfa++; continue; }
      if (x < 0) x = -x;
      if (x > maxaa) maxaa = x;
      double y = bb[k]; if (!isfinite(y)) { nfb++; continue; }
      if (y < 0) y = -y;
      if (y > maxbb) maxbb = y;
    }
    int64_t first_nonfin = -1;
    double v = aa[(N - 1) * N];
    for (int64_t i = 1; i < N; i++) {
      double t = v * v;
      v = t + bb[(N - 1) * N + i];
      if (!isfinite(v)) { first_nonfin = i; break; }
    }
    printf("PROBE N=%lld maxaa=%.3g maxbb=%.3g nonfin_aa=%d nonfin_bb=%d longest_row_nonfin_at=%lld\n",
           (long long)N, maxaa, maxbb, nfa, nfb, (long long)first_nonfin);
    fflush(stdout);
  }

  const int64_t nrows = LEN_2D;
  int nt = (int)omp_get_max_threads();
  if (nt > (int)(nrows - 1)) nt = (int)(nrows - 1);
  if (nt < 1) nt = 1;

  static int64_t starts[8193];
  for (int t = 0; t <= nt; t++)
    starts[t] = (int64_t)((double)nrows * sqrt((double)t / (double)nt) + 0.5);

  #pragma omp parallel num_threads(nt)
  {
    const int tid = omp_get_thread_num();
    for (int64_t j = starts[tid]; j < starts[tid + 1]; ++j) {
      double *restrict a = aa + j * LEN_2D;
      const double *restrict b = bb + j * LEN_2D;
      double v = a[0];
      for (int64_t i = 1; i <= j; ++i) {
        v = v * v + b[i];
        __asm__ volatile("movntsd %1, %0" : "=m"(a[i]) : "x"(v));
      }
    }
    _mm_sfence();
  }
}
