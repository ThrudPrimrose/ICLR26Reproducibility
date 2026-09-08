#include <stdint.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <immintrin.h>

static void body512(double *restrict a, const double *restrict b, int64_t n, double ds, int nt, int pf) {
  __m512d vds = _mm512_set1_pd(ds);
  int64_t i = 0;
  while (i < n && ((uintptr_t)(a+i) & 31)) { a[i] += b[i]*ds; i++; }
  int64_t nv = (n - i) / 8;
  for (int64_t k = 0; k < nv; k++) {
    if (pf) {
      int64_t d = (pf==1 ? 16 : 32) * 8;
      if (i + k*8 + d < n) {
        _mm_prefetch((const char*)(b + i + k*8 + d), _MM_HINT_T0);
        _mm_prefetch((const char*)(a + i + k*8 + d), _MM_HINT_T0);
      }
    }
    __m512d va = _mm512_loadu_pd(a + i + k*8);
    __m512d vb = _mm512_loadu_pd(b + i + k*8);
    if (nt) _mm512_stream_pd(a + i + k*8, _mm512_fmadd_pd(va, vb, vds));
    else    _mm512_storeu_pd(a + i + k*8, _mm512_fmadd_pd(va, vb, vds));
  }
  i += nv*8;
  for (; i < n; i++) a[i] += b[i]*ds;
}

static double run_variant(double *restrict a, const double *restrict b, int64_t n, double ds, int nt, int pf) {
  const int64_t CHUNK = 256;
  int64_t nch = (n + CHUNK - 1) / CHUNK;
  double best = 0;
  for (int rep = 0; rep < 3; rep++) {
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    #pragma omp parallel for schedule(static)
    for (int64_t c = 0; c < nch; c++) {
      int64_t off = c*CHUNK;
      int64_t len = CHUNK < n - off ? CHUNK : n - off;
      body512(a+off, b+off, len, ds, nt, pf);
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double dt;
    dt = (double)(t1.tv_sec-t0.tv_sec) + 1e-9*(double)(t1.tv_nsec-t0.tv_nsec);
    double bw = (double)n*24.0/dt/1e9; if (bw>best) best=bw;
  }
  return best;
}

static void print_numa(const char *tag, const void *p) {
  FILE *f = fopen("/proc/self/maps","r");
  if (!f) return;
  uintptr_t addr = (uintptr_t)p, vs = 0;
  char line[512];
  while (fgets(line,512,f)) {
    unsigned long s,e; if (sscanf(line,"%lx-%lx",&s,&e)==2) { if ((uintptr_t)s<=addr && addr<(uintptr_t)e) { vs=s; break; } }
  }
  fclose(f);
  if (!vs) { fprintf(stdout,"D %s maps: not found\n",tag); return; }
  f = fopen("/proc/self/numa_maps","r");
  if (!f) return;
  while (fgets(line,512,f)) {
    unsigned long s,e; if (sscanf(line,"%lx-%lx",&s,&e)==2 && s==vs) {
      char out[200]; snprintf(out,200,"D %s numa: %.180s",tag,line);
      fprintf(stdout,"%s\n",out); break;
    }
  }
  fclose(f);
}

void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D, const int64_t S) {
  const double ds = (double)S;
  print_numa("a", a);
  print_numa("b", b);
  /* plain (auto-vectorized) */
  {
    double best = 0;
    for (int rep = 0; rep < 3; rep++) {
      struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC,&t0);
      #pragma omp parallel for schedule(static)
      for (int64_t i = 0; i < LEN_1D; i++) a[i] += b[i]*ds;
      clock_gettime(CLOCK_MONOTONIC,&t1);
      double dt = (double)(t1.tv_sec-t0.tv_sec) + 1e-9*(double)(t1.tv_nsec-t0.tv_nsec);
      double bw = (double)LEN_1D*24.0/dt/1e9; if (bw>best) best=bw;
    }
    fprintf(stdout, "D plain   24t=%.2f GB/s\n", best);
  }
  fprintf(stdout, "D avx512  24t=%.2f GB/s\n", run_variant(a,b,LEN_1D,ds,0,0));
  fprintf(stdout, "D ntstore 24t=%.2f GB/s\n", run_variant(a,b,LEN_1D,ds,1,0));
  fprintf(stdout, "D pf16    24t=%.2f GB/s\n", run_variant(a,b,LEN_1D,ds,0,1));
  fprintf(stdout, "D pf32    24t=%.2f GB/s\n", run_variant(a,b,LEN_1D,ds,0,2));
  fprintf(stdout, "D nt+pf16 24t=%.2f GB/s\n", run_variant(a,b,LEN_1D,ds,1,1));
  fflush(stdout);
}
