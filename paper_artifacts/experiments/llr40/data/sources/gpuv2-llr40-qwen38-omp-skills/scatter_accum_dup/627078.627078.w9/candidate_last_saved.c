#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void scatter_accum_dup_fp64(double *restrict bins, const double *restrict src,
                            const int32_t *restrict ip, const int64_t LEN_1D) {
  static int diag = 0;
  if (!diag) {
    diag = 1;
    printf("DIAG LEN=%lld bins=%p src=%p ip=%p\n", (long long)LEN_1D,
           (void *)bins, (void *)src, (void *)ip);
    const char *es[] = {"OMP_NUM_DEVICES","OMP_TARGET_OFFLOAD","OMP_NUM_THREADS",
      "HSA_ENABLE_SDMA","HSA_ENABLE_DMA","HSA_MAX_COPY_SIZE","LIBOMPTARGET_INFO",
      "LIBOMPTARGET_VERBOSE","OMP_MAX_ACTIVE_LEVELS","OMP_PLACES","OMP_PROC_BIND",
      "HSA_XNACK","HSA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","CUDA_VISIBLE_DEVICES","CUDA_MPS_PIPE_DIRECTORY"};
    for (unsigned k = 0; k < sizeof(es)/sizeof(es[0]); ++k) {
      const char *v = getenv(es[k]);
      printf("ENV %s=%s\n", es[k], v ? v : "(null)");
    }
    fflush(stdout);
  }
  if (LEN_1D <= 0) return;
  #pragma omp target data \
    map(tofrom: bins[0:LEN_1D]) map(to: src[0:LEN_1D]) map(to: ip[0:LEN_1D])
  {
    printf("DIAG stage1: HtoD done\n"); fflush(stdout);
    #pragma omp target teams distribute parallel for
    for (int64_t i = 0; i < LEN_1D; ++i) {
      #pragma omp atomic update
      bins[ip[i]] += src[i];
    }
    printf("DIAG stage2: compute kernel done\n"); fflush(stdout);
  }
  printf("DIAG stage3: DtoH done\n"); fflush(stdout);
}
