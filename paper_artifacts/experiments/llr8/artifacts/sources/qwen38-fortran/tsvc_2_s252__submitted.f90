subroutine tsvc_2_s252_fp64(a, b, c, n) bind(C, name='tsvc_2_s252_fp64')
  use, intrinsic :: iso_c_binding
  use, intrinsic :: omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: n
  real(c_double), intent(out) :: a(n)
  real(c_double), intent(in) :: b(n), c(n)
  integer(c_int64_t) :: i, k, lo, hi, base, rem, size_k
  integer(c_int64_t) :: nt
  real(c_double) :: s, t

  if (n == 0) return
  a(1) = b(1) * c(1)
  if (n == 1) return

  !$omp parallel default(none) shared(a, b, c, n) private(nt, k, lo, hi, base, rem, size_k, i, s, t)
  nt = int(omp_get_num_threads(), c_int64_t)
  k = omp_get_thread_num()
  base = (n - 1_c_int64_t) / nt
  rem = mod(n - 1_c_int64_t, nt)
  size_k = base + merge(1_c_int64_t, 0_c_int64_t, k < rem)
  lo = 2_c_int64_t + k * base + min(k, rem)
  hi = lo + size_k - 1_c_int64_t
  if (lo <= hi) then
    t = b(lo - 1_c_int64_t) * c(lo - 1_c_int64_t)
    do i = lo, hi
      s = b(i) * c(i)
      a(i) = s + t
      t = s
    end do
  end if
  !$omp end parallel
end subroutine tsvc_2_s252_fp64
