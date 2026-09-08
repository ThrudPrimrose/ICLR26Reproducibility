#include <stdint.h>
#include <omp.h>

/* madvise(addr,len,MADV_HUGEPAGE) via raw syscall (feature-test macro hides madvise).
   Syscall 28 on x86-64: rdi=addr rsi=len rdx=advice. */
static inline void madvise_hp(void *addr, unsigned long len) {
  long ret;
  __asm__ volatile ("syscall"
    : "=a"(ret)
    : "0"(28L), "D"(addr), "S"(len), "d"(4L)
    : "rcx", "r11", "memory");
  (void)ret;
}

void ext_war_unit_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
  if (LEN_1D < 2) return;
  if (LEN_1D >= 1000000) {
    const unsigned long sz = (unsigned long)LEN_1D * sizeof(double);
    madvise_hp(a, sz);
    madvise_hp((void *)b, sz);
  }
  if (LEN_1D < 1000000) {
    for (int64_t i = 0; i < LEN_1D - 1; ++i)
      a[i] = a[i + 1] + b[i];
    return;
  }
  const int64_t n = LEN_1D - 1;
  #pragma omp parallel
  {
    const int t = omp_get_thread_num();
    const int m = omp_get_num_threads();
    const int64_t base = n / m, rem = n % m;
    const int64_t L = t * base + (t < rem ? t : rem);
    const int64_t R = (t + 1) * base + (t + 1 < rem ? t + 1 : rem);
    const double save = a[R];
    #pragma omp barrier
    if (L < R) {
      int64_t i;
      for (i = L; i < R - 1; ++i)
        a[i] = a[i + 1] + b[i];
      a[R - 1] = save + b[R - 1];
    }
  }
}
