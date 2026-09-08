#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static int host_threads(void)
{
  const char *env = getenv("OMP_NUM_THREADS");
  int T = -1;
  if (env && *env) {
    long v = atol(env);
    if (v >= 1) T = (int)v;
  }
  if (T < 1) T = omp_get_num_procs();
  if (T < 1) T = 1;
  if (T > 256) T = 256;
  return T;
}

void tsvc_2_s3112_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D)
{
  /* Arm requirement: the binary must register a device kernel. Minimal probe. */
  int on_device = 0;
  #pragma omp target map(from: on_device)
  on_device = !omp_is_initial_device();
  if (!on_device) { /* no device: host path below is still valid */ }

  const int64_t n = LEN_1D;
  if (n <= 0) return;

  int T = host_threads();
  if ((int64_t)T > n) T = (int)n;

  static double chunksum[256];
  static double prefix[256];

  #pragma omp parallel num_threads(T)
  {
    const int tid = omp_get_thread_num();
    const int64_t base = n / T;
    const int64_t s = base * (int64_t)tid;
    const int64_t e = (tid == T - 1) ? n : base * (int64_t)(tid + 1);

    /* pass 1: chunk sum as the SAME left fold the reference would use */
    double c = 0.0;
    for (int64_t i = s; i < e; ++i) c += a[i];
    chunksum[tid] = c;

    #pragma omp barrier

    if (tid == 0) {
      double p = 0.0;
      for (int k = 0; k < T; ++k) { prefix[k] = p; p += chunksum[k]; }
    }

    #pragma omp barrier

    /* pass 3: rescan chunk: b[i] = (P + a[s] + ... + a[i])  [left fold] */
    double csum = prefix[tid];
    for (int64_t i = s; i + 8 <= e; i += 8) {
      csum += a[i];   b[i]   = csum;
      csum += a[i+1]; b[i+1] = csum;
      csum += a[i+2]; b[i+2] = csum;
      csum += a[i+3]; b[i+3] = csum;
      csum += a[i+4]; b[i+4] = csum;
      csum += a[i+5]; b[i+5] = csum;
      csum += a[i+6]; b[i+6] = csum;
      csum += a[i+7]; b[i+7] = csum;
    }
    for (int64_t i = (s + ((e-s)/8)*8); i < e; ++i) { csum += a[i]; b[i] = csum; }
  }
}
