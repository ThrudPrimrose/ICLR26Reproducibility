#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern long syscall(long, ...);
#define MY_MINCORE 27
#define MY_SCHGAFF 204
#define MY_SCHSETAFF 203
#define MY_GETMEMPOLICY 238
#define MPOL_F_ADDR 0x01

typedef unsigned long mycpu_set_t[128];

static void mempol(const double *p, const char *name) {
  int mode = 0;
  unsigned long nm[8] = {0};
  long r = syscall(MY_GETMEMPOLICY, &mode, nm, (unsigned long)32, (void*)p, (unsigned long)MPOL_F_ADDR);
  if (r >= 0) printf("%s: mempol mode=%d mask=", name, mode);
  else printf("%s: mempol failed\n", name);
}

static void prfile(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) { printf("open %s FAILED\n", path); return; }
  char line[512];
  while (fgets(line, sizeof(line), f)) {
    if (strstr(line, "Cpus_allowed") || strstr(line, "Mems_allowed")) { printf("%s: %s", path, line); }
  }
  fclose(f);
}

void tsvc_2_s2710_fp64(double *restrict a, double *restrict b, double *restrict c, const double *restrict d,
                       const double *restrict e, const double *restrict x, const int64_t LEN_1D) {
  static int once = 0;
  if (!once) {
    once = 1;
    printf("LEN=%lld avx512f=%d\n", (long long)LEN_1D, (int)__builtin_cpu_supports("avx512f"));
    prfile("/proc/self/status");
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f) { char line[512]; int shown=0; while (fgets(line,sizeof(line),f)) { if (strncmp(line,"model name",10)==0 && !shown) { printf("cpuinfo: %s", line); shown=1; } if (strncmp(line,"processor",9)==0) shown++; } printf("cpuinfo: %d processors\n", shown); fclose(f); }
    mycpu_set_t cs;
    memset(cs, 0, sizeof(cs));
    long r = syscall(MY_SCHGAFF, 0, (size_t)sizeof(cs), cs);
    printf("getaffinity r=%ld\n", r);
    if (r == 0) {
      int first = -1;
      for (int i = 0; i < 1024; i++) if ((cs[i/64] >> (i%64)) & 1UL) { first = i; break; }
      printf("first_cpu=%d\n", first);
      mycpu_set_t one; memset(one, 0, sizeof(one));
      if (first >= 0) one[first/64] = 1UL << (first%64);
      long s = syscall(MY_SCHSETAFF, 0, (size_t)sizeof(one), one);
      printf("setaffinity r=%ld first=%d\n", s, first);
      if (s == 0) { long s2 = syscall(MY_SCHSETAFF, 0, (size_t)sizeof(cs), cs); printf("setaffinity-restore r=%ld\n", s2); }
    }
    mempol(a, "a"); mempol(b, "b"); mempol(c, "c"); mempol(d, "d"); mempol(e, "e"); mempol(x, "x");
    fflush(stdout);
  }
  for (int64_t i = 0; i < LEN_1D; ++i) {
    if (a[i] > b[i]) {
      a[i] += b[i] * d[i];
      if (LEN_1D > 10) { c[i] += d[i] * d[i]; } else { c[i] = d[i] * e[i] + 1.0; }
    } else {
      b[i] = a[i] + e[i] * e[i];
      if (x[0] > 0.0) { c[i] = a[i] + d[i] * d[i]; } else { c[i] += e[i] * e[i]; }
    }
  }
}
